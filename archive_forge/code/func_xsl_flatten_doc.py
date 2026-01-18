from pathlib import Path
import pytest
@pytest.fixture
def xsl_flatten_doc(xml_data_path, datapath):
    return datapath(xml_data_path / 'flatten_doc.xsl')