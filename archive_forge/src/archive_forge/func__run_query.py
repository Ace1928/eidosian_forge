from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _run_query(self) -> List[Dict[str, Any]]:
    try:
        import oracledb
    except ImportError as e:
        raise ImportError("Could not import oracledb, please install with 'pip install oracledb'") from e
    connect_param = {'user': self.user, 'password': self.password, 'dsn': self.dsn}
    if self.dsn == self.tns_name:
        connect_param['config_dir'] = self.config_dir
    if self.wallet_location and self.wallet_password:
        connect_param['wallet_location'] = self.wallet_location
        connect_param['wallet_password'] = self.wallet_password
    try:
        connection = oracledb.connect(**connect_param)
        cursor = connection.cursor()
        if self.schema:
            cursor.execute(f'alter session set current_schema={self.schema}')
        cursor.execute(self.query)
        columns = [col[0] for col in cursor.description]
        data = cursor.fetchall()
        data = [dict(zip(columns, row)) for row in data]
    except oracledb.DatabaseError as e:
        print('Got error while connecting: ' + str(e))
        data = []
    finally:
        cursor.close()
        connection.close()
    return data