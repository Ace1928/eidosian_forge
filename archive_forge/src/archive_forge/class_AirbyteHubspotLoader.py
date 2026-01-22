from typing import Any, Callable, Iterator, Mapping, Optional
from langchain_core.documents import Document
from langchain_core.utils.utils import guard_import
from langchain_community.document_loaders.base import BaseLoader
class AirbyteHubspotLoader(AirbyteCDKLoader):
    """Load from `Hubspot` using an `Airbyte` source connector."""

    def __init__(self, config: Mapping[str, Any], stream_name: str, record_handler: Optional[RecordHandler]=None, state: Optional[Any]=None) -> None:
        """Initializes the loader.

        Args:
            config: The config to pass to the source connector.
            stream_name: The name of the stream to load.
            record_handler: A function that takes in a record and an optional id and
                returns a Document. If None, the record will be used as the document.
                Defaults to None.
            state: The state to pass to the source connector. Defaults to None.
        """
        source_class = guard_import('source_hubspot', pip_name='airbyte-source-hubspot').SourceHubspot
        super().__init__(config=config, source_class=source_class, stream_name=stream_name, record_handler=record_handler, state=state)