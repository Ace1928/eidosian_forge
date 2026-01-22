import asyncio
from collections import defaultdict
from collections.abc import Hashable
from itertools import chain
from typing import (
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.load.dump import dumpd
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config, patch_config
from langchain_core.runnables.utils import (
class EnsembleRetriever(BaseRetriever):
    """Retriever that ensembles the multiple retrievers.

    It uses a rank fusion.

    Args:
        retrievers: A list of retrievers to ensemble.
        weights: A list of weights corresponding to the retrievers. Defaults to equal
            weighting for all retrievers.
        c: A constant added to the rank, controlling the balance between the importance
            of high-ranked items and the consideration given to lower-ranked items.
            Default is 60.
    """
    retrievers: List[RetrieverLike]
    weights: List[float]
    c: int = 60

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        """List configurable fields for this runnable."""
        return get_unique_config_specs((spec for retriever in self.retrievers for spec in retriever.config_specs))

    @root_validator(pre=True)
    def set_weights(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values.get('weights'):
            n_retrievers = len(values['retrievers'])
            values['weights'] = [1 / n_retrievers] * n_retrievers
        return values

    def invoke(self, input: str, config: Optional[RunnableConfig]=None, **kwargs: Any) -> List[Document]:
        from langchain_core.callbacks import CallbackManager
        config = ensure_config(config)
        callback_manager = CallbackManager.configure(config.get('callbacks'), None, verbose=kwargs.get('verbose', False), inheritable_tags=config.get('tags', []), local_tags=self.tags, inheritable_metadata=config.get('metadata', {}), local_metadata=self.metadata)
        run_manager = callback_manager.on_retriever_start(dumpd(self), input, name=config.get('run_name'), **kwargs)
        try:
            result = self.rank_fusion(input, run_manager=run_manager, config=config)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e
        else:
            run_manager.on_retriever_end(result, **kwargs)
            return result

    async def ainvoke(self, input: str, config: Optional[RunnableConfig]=None, **kwargs: Any) -> List[Document]:
        from langchain_core.callbacks import AsyncCallbackManager
        config = ensure_config(config)
        callback_manager = AsyncCallbackManager.configure(config.get('callbacks'), None, verbose=kwargs.get('verbose', False), inheritable_tags=config.get('tags', []), local_tags=self.tags, inheritable_metadata=config.get('metadata', {}), local_metadata=self.metadata)
        run_manager = await callback_manager.on_retriever_start(dumpd(self), input, name=config.get('run_name'), **kwargs)
        try:
            result = await self.arank_fusion(input, run_manager=run_manager, config=config)
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise e
        else:
            await run_manager.on_retriever_end(result, **kwargs)
            return result

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        Get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """
        fused_documents = self.rank_fusion(query, run_manager)
        return fused_documents

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
        """
        Asynchronously get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """
        fused_documents = await self.arank_fusion(query, run_manager)
        return fused_documents

    def rank_fusion(self, query: str, run_manager: CallbackManagerForRetrieverRun, *, config: Optional[RunnableConfig]=None) -> List[Document]:
        """
        Retrieve the results of the retrievers and use rank_fusion_func to get
        the final result.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """
        retriever_docs = [retriever.invoke(query, patch_config(config, callbacks=run_manager.get_child(tag=f'retriever_{i + 1}'))) for i, retriever in enumerate(self.retrievers)]
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [Document(page_content=cast(str, doc)) if isinstance(doc, str) else doc for doc in retriever_docs[i]]
        fused_documents = self.weighted_reciprocal_rank(retriever_docs)
        return fused_documents

    async def arank_fusion(self, query: str, run_manager: AsyncCallbackManagerForRetrieverRun, *, config: Optional[RunnableConfig]=None) -> List[Document]:
        """
        Asynchronously retrieve the results of the retrievers
        and use rank_fusion_func to get the final result.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """
        retriever_docs = await asyncio.gather(*[retriever.ainvoke(query, patch_config(config, callbacks=run_manager.get_child(tag=f'retriever_{i + 1}'))) for i, retriever in enumerate(self.retrievers)])
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [Document(page_content=doc) if not isinstance(doc, Document) else doc for doc in retriever_docs[i]]
        fused_documents = self.weighted_reciprocal_rank(retriever_docs)
        return fused_documents

    def weighted_reciprocal_rank(self, doc_lists: List[List[Document]]) -> List[Document]:
        """
        Perform weighted Reciprocal Rank Fusion on multiple rank lists.
        You can find more details about RRF here:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.

        Returns:
            list: The final aggregated list of items sorted by their weighted RRF
                    scores in descending order.
        """
        if len(doc_lists) != len(self.weights):
            raise ValueError('Number of rank lists must be equal to the number of weights.')
        rrf_score: Dict[str, float] = defaultdict(float)
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[doc.page_content] += weight / (rank + self.c)
        all_docs = chain.from_iterable(doc_lists)
        sorted_docs = sorted(unique_by_key(all_docs, lambda doc: doc.page_content), reverse=True, key=lambda doc: rrf_score[doc.page_content])
        return sorted_docs