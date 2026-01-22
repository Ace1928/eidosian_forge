import re
from abc import ABC, abstractmethod
from typing import (
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from typing_extensions import Annotated
class AmazonKendraRetriever(BaseRetriever):
    """`Amazon Kendra Index` retriever.

    Args:
        index_id: Kendra index id

        region_name: The aws region e.g., `us-west-2`.
            Fallsback to AWS_DEFAULT_REGION env variable
            or region specified in ~/.aws/config.

        credentials_profile_name: The name of the profile in the ~/.aws/credentials
            or ~/.aws/config files, which has either access keys or role information
            specified. If not specified, the default credential profile or, if on an
            EC2 instance, credentials from IMDS will be used.

        top_k: No of results to return

        attribute_filter: Additional filtering of results based on metadata
            See: https://docs.aws.amazon.com/kendra/latest/APIReference

        page_content_formatter: generates the Document page_content
            allowing access to all result item attributes. By default, it uses
            the item's title and excerpt.

        client: boto3 client for Kendra

        user_context: Provides information about the user context
            See: https://docs.aws.amazon.com/kendra/latest/APIReference

    Example:
        .. code-block:: python

            retriever = AmazonKendraRetriever(
                index_id="c0806df7-e76b-4bce-9b5c-d5582f6b1a03"
            )

    """
    index_id: str
    region_name: Optional[str] = None
    credentials_profile_name: Optional[str] = None
    top_k: int = 3
    attribute_filter: Optional[Dict] = None
    page_content_formatter: Callable[[ResultItem], str] = combined_text
    client: Any
    user_context: Optional[Dict] = None
    min_score_confidence: Annotated[Optional[float], Field(ge=0.0, le=1.0)]

    @validator('top_k')
    def validate_top_k(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f'top_k ({value}) cannot be negative.')
        return value

    @root_validator(pre=True)
    def create_client(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get('client') is not None:
            return values
        try:
            import boto3
            if values.get('credentials_profile_name'):
                session = boto3.Session(profile_name=values['credentials_profile_name'])
            else:
                session = boto3.Session()
            client_params = {}
            if values.get('region_name'):
                client_params['region_name'] = values['region_name']
            values['client'] = session.client('kendra', **client_params)
            return values
        except ImportError:
            raise ModuleNotFoundError('Could not import boto3 python package. Please install it with `pip install boto3`.')
        except Exception as e:
            raise ValueError('Could not load credentials to authenticate with AWS client. Please check that credentials in the specified profile name are valid.') from e

    def _kendra_query(self, query: str) -> Sequence[ResultItem]:
        kendra_kwargs = {'IndexId': self.index_id, 'QueryText': query.strip()[0:999], 'PageSize': self.top_k}
        if self.attribute_filter is not None:
            kendra_kwargs['AttributeFilter'] = self.attribute_filter
        if self.user_context is not None:
            kendra_kwargs['UserContext'] = self.user_context
        response = self.client.retrieve(**kendra_kwargs)
        r_result = RetrieveResult.parse_obj(response)
        if r_result.ResultItems:
            return r_result.ResultItems
        response = self.client.query(**kendra_kwargs)
        q_result = QueryResult.parse_obj(response)
        return q_result.ResultItems

    def _get_top_k_docs(self, result_items: Sequence[ResultItem]) -> List[Document]:
        top_docs = [item.to_doc(self.page_content_formatter) for item in result_items[:self.top_k]]
        return top_docs

    def _filter_by_score_confidence(self, docs: List[Document]) -> List[Document]:
        """
        Filter out the records that have a score confidence
        greater than the required threshold.
        """
        if not self.min_score_confidence:
            return docs
        filtered_docs = [item for item in docs if item.metadata.get('score') is not None and isinstance(item.metadata['score'], str) and (KENDRA_CONFIDENCE_MAPPING.get(item.metadata['score'], 0.0) >= self.min_score_confidence)]
        return filtered_docs

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Run search on Kendra index and get top k documents

        Example:
        .. code-block:: python

            docs = retriever.get_relevant_documents('This is my query')

        """
        result_items = self._kendra_query(query)
        top_k_docs = self._get_top_k_docs(result_items)
        return self._filter_by_score_confidence(top_k_docs)