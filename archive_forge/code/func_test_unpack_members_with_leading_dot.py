from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
import pytest
from .. import Pooch
from ..processors import Unzip, Untar, Decompress
from .utils import pooch_test_url, pooch_test_registry, check_tiny_data, capture_log
@pytest.mark.network
@pytest.mark.parametrize('processor_class,extension', [(Unzip, '.zip'), (Untar, '.tar.gz')])
def test_unpack_members_with_leading_dot(processor_class, extension):
    """Test that unpack members can also be specifed both with a leading ./"""
    with TemporaryDirectory() as local_store:
        pup = Pooch(path=Path(local_store), base_url=BASEURL, registry=REGISTRY)
        processor1 = processor_class(members=['./store/tiny-data.txt'])
        filenames1 = pup.fetch('store' + extension, processor=processor1)
        assert len(filenames1) == 1
        check_tiny_data(filenames1[0])