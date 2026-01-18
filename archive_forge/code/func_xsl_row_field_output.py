from pathlib import Path
import pytest
@pytest.fixture
def xsl_row_field_output(xml_data_path, datapath):
    return datapath(xml_data_path / 'row_field_output.xsl')