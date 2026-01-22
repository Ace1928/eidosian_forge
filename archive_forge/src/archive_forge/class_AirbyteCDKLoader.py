from typing import Any, Callable, Iterator, Mapping, Optional
from langchain_core.documents import Document
from langchain_core.utils.utils import guard_import
from langchain_community.document_loaders.base import BaseLoader
class AirbyteCDKLoader(BaseLoader):
    """Load with an `Airbyte` source connector implemented using the `CDK`."""

    def __init__(self, config: Mapping[str, Any], source_class: Any, stream_name: str, record_handler: Optional[RecordHandler]=None, state: Optional[Any]=None) -> None:
        """Initializes the loader.

        Args:
            config: The config to pass to the source connector.
            source_class: The source connector class.
            stream_name: The name of the stream to load.
            record_handler: A function that takes in a record and an optional id and
                returns a Document. If None, the record will be used as the document.
                Defaults to None.
            state: The state to pass to the source connector. Defaults to None.
        """
        from airbyte_cdk.models.airbyte_protocol import AirbyteRecordMessage
        from airbyte_cdk.sources.embedded.base_integration import BaseEmbeddedIntegration
        from airbyte_cdk.sources.embedded.runner import CDKRunner

        class CDKIntegration(BaseEmbeddedIntegration):
            """A wrapper around the CDK integration."""

            def _handle_record(self, record: AirbyteRecordMessage, id: Optional[str]) -> Document:
                if record_handler:
                    return record_handler(record, id)
                return Document(page_content='', metadata=record.data)
        self._integration = CDKIntegration(config=config, runner=CDKRunner(source=source_class(), name=source_class.__name__))
        self._stream_name = stream_name
        self._state = state

    def lazy_load(self) -> Iterator[Document]:
        return self._integration._load_data(stream_name=self._stream_name, state=self._state)

    @property
    def last_state(self) -> Any:
        return self._integration.last_state