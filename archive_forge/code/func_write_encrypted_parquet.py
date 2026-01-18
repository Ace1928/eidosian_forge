import pytest
from datetime import timedelta
import pyarrow as pa
def write_encrypted_parquet(path, table, encryption_config, kms_connection_config, crypto_factory):
    file_encryption_properties = crypto_factory.file_encryption_properties(kms_connection_config, encryption_config)
    assert file_encryption_properties is not None
    with pq.ParquetWriter(path, table.schema, encryption_properties=file_encryption_properties) as writer:
        writer.write_table(table)