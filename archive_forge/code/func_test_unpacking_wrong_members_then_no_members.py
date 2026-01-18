from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
import pytest
from .. import Pooch
from ..processors import Unzip, Untar, Decompress
from .utils import pooch_test_url, pooch_test_registry, check_tiny_data, capture_log
@pytest.mark.network
@pytest.mark.parametrize('processor_class,extension', [(Unzip, '.zip'), (Untar, '.tar.gz')])
def test_unpacking_wrong_members_then_no_members(processor_class, extension):
    """
    Test that calling with invalid members then without them works.
    https://github.com/fatiando/pooch/issues/364
    """
    with TemporaryDirectory() as local_store:
        pup = Pooch(path=Path(local_store), base_url=BASEURL, registry=REGISTRY)
        processor1 = processor_class(members=['not-a-valid-file.csv'])
        filenames1 = pup.fetch('store' + extension, processor=processor1)
        assert len(filenames1) == 0
        processor2 = processor_class()
        filenames2 = pup.fetch('store' + extension, processor=processor2)
        assert len(filenames2) > 0