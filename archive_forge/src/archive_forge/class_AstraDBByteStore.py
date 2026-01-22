from __future__ import annotations
import base64
from abc import ABC, abstractmethod
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.stores import BaseStore, ByteStore
from langchain_community.utilities.astradb import (
@deprecated(since='0.0.22', removal='0.2.0', alternative_import='langchain_astradb.AstraDBByteStore')
class AstraDBByteStore(AstraDBBaseStore[bytes], ByteStore):

    def __init__(self, collection_name: str, token: Optional[str]=None, api_endpoint: Optional[str]=None, astra_db_client: Optional[AstraDB]=None, namespace: Optional[str]=None, *, async_astra_db_client: Optional[AsyncAstraDB]=None, pre_delete_collection: bool=False, setup_mode: SetupMode=SetupMode.SYNC) -> None:
        """ByteStore implementation using DataStax AstraDB as the underlying store.

        The bytes values are converted to base64 encoded strings
        Documents in the AstraDB collection will have the format

        .. code-block:: json

            {
              "_id": "<key>",
              "value": "<byte64 string value>"
            }

        Args:
            collection_name: name of the Astra DB collection to create/use.
            token: API token for Astra DB usage.
            api_endpoint: full URL to the API endpoint,
                such as `https://<DB-ID>-us-east1.apps.astra.datastax.com`.
            astra_db_client: *alternative to token+api_endpoint*,
                you can pass an already-created 'astrapy.db.AstraDB' instance.
            async_astra_db_client: *alternative to token+api_endpoint*,
                you can pass an already-created 'astrapy.db.AsyncAstraDB' instance.
            namespace: namespace (aka keyspace) where the
                collection is created. Defaults to the database's "default namespace".
            setup_mode: mode used to create the Astra DB collection (SYNC, ASYNC or
                OFF).
            pre_delete_collection: whether to delete the collection
                before creating it. If False and the collection already exists,
                the collection will be used as is.
        """
        super().__init__(collection_name=collection_name, token=token, api_endpoint=api_endpoint, astra_db_client=astra_db_client, async_astra_db_client=async_astra_db_client, namespace=namespace, setup_mode=setup_mode, pre_delete_collection=pre_delete_collection)

    def decode_value(self, value: Any) -> Optional[bytes]:
        if value is None:
            return None
        return base64.b64decode(value)

    def encode_value(self, value: Optional[bytes]) -> Any:
        if value is None:
            return None
        return base64.b64encode(value).decode('ascii')