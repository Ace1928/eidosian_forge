from __future__ import annotations
import logging
from typing import Any, Iterable, List, Optional, Tuple, Union
from uuid import uuid4
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _create_connection_alias(self, connection_args: dict) -> str:
    """Create the connection to the Milvus server."""
    from pymilvus import MilvusException, connections
    host: str = connection_args.get('host', None)
    port: Union[str, int] = connection_args.get('port', None)
    address: str = connection_args.get('address', None)
    uri: str = connection_args.get('uri', None)
    user = connection_args.get('user', None)
    if host is not None and port is not None:
        given_address = str(host) + ':' + str(port)
    elif uri is not None:
        if uri.startswith('https://'):
            given_address = uri.split('https://')[1]
        elif uri.startswith('http://'):
            given_address = uri.split('http://')[1]
        else:
            logger.error('Invalid Milvus URI: %s', uri)
            raise ValueError('Invalid Milvus URI: %s', uri)
    elif address is not None:
        given_address = address
    else:
        given_address = None
        logger.debug('Missing standard address type for reuse attempt')
    if user is not None:
        tmp_user = user
    else:
        tmp_user = ''
    if given_address is not None:
        for con in connections.list_connections():
            addr = connections.get_connection_addr(con[0])
            if con[1] and 'address' in addr and (addr['address'] == given_address) and ('user' in addr) and (addr['user'] == tmp_user):
                logger.debug('Using previous connection: %s', con[0])
                return con[0]
    alias = uuid4().hex
    try:
        connections.connect(alias=alias, **connection_args)
        logger.debug('Created new connection using: %s', alias)
        return alias
    except MilvusException as e:
        logger.error('Failed to create new connection using: %s', alias)
        raise e