from pathlib import Path
import pytest
@pytest.fixture
def xml_books(xml_data_path, datapath):
    return datapath(xml_data_path / 'books.xml')