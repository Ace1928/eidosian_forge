from pathlib import Path
import pytest
@pytest.fixture
def xml_doc_ch_utf(xml_data_path, datapath):
    return datapath(xml_data_path / 'doc_ch_utf.xml')