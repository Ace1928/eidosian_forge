from typing import Any, Dict, Optional, Sequence, Type, Union
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
class OpenAIMetadataTagger(BaseDocumentTransformer, BaseModel):
    """Extract metadata tags from document contents using OpenAI functions.

    Example:
        .. code-block:: python

                from langchain_community.chat_models import ChatOpenAI
                from langchain_community.document_transformers import OpenAIMetadataTagger
                from langchain_core.documents import Document

                schema = {
                    "properties": {
                        "movie_title": { "type": "string" },
                        "critic": { "type": "string" },
                        "tone": {
                            "type": "string",
                            "enum": ["positive", "negative"]
                        },
                        "rating": {
                            "type": "integer",
                            "description": "The number of stars the critic rated the movie"
                        }
                    },
                    "required": ["movie_title", "critic", "tone"]
                }

                # Must be an OpenAI model that supports functions
                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
                tagging_chain = create_tagging_chain(schema, llm)
                document_transformer = OpenAIMetadataTagger(tagging_chain=tagging_chain)
                original_documents = [
                    Document(page_content="Review of The Bee Movie
By Roger Ebert

This is the greatest movie ever made. 4 out of 5 stars."),
                    Document(page_content="Review of The Godfather
By Anonymous

This movie was super boring. 1 out of 5 stars.", metadata={"reliable": False}),
                ]

                enhanced_documents = document_transformer.transform_documents(original_documents)
    """
    tagging_chain: Any
    'The chain used to extract metadata from each document.'

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """Automatically extract and populate metadata
        for each document according to the provided schema."""
        new_documents = []
        for document in documents:
            extracted_metadata: Dict = self.tagging_chain.run(document.page_content)
            new_document = Document(page_content=document.page_content, metadata={**extracted_metadata, **document.metadata})
            new_documents.append(new_document)
        return new_documents

    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        raise NotImplementedError