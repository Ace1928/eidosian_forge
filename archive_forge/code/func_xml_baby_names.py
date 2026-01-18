from pathlib import Path
import pytest
@pytest.fixture
def xml_baby_names(xml_data_path, datapath):
    return datapath(xml_data_path / 'baby_names.xml')