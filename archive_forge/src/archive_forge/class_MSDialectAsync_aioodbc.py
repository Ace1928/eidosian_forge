from __future__ import annotations
from .pyodbc import MSDialect_pyodbc
from .pyodbc import MSExecutionContext_pyodbc
from ...connectors.aioodbc import aiodbcConnector
class MSDialectAsync_aioodbc(aiodbcConnector, MSDialect_pyodbc):
    driver = 'aioodbc'
    supports_statement_cache = True
    execution_ctx_cls = MSExecutionContext_aioodbc