from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
import pytest
from .. import Pooch
from ..processors import Unzip, Untar, Decompress
from .utils import pooch_test_url, pooch_test_registry, check_tiny_data, capture_log
@pytest.mark.network
@pytest.mark.parametrize('target_path', [None, 'some_custom_path'], ids=['default_path', 'custom_path'])
@pytest.mark.parametrize('archive,members', [('tiny-data', ['tiny-data.txt']), ('store', None), ('store', ['store/tiny-data.txt']), ('store', ['store/subdir/tiny-data.txt']), ('store', ['store/subdir']), ('store', ['store/tiny-data.txt', 'store/subdir'])], ids=['single_file', 'archive_all', 'archive_file', 'archive_subdir_file', 'archive_subdir', 'archive_multiple'])
@pytest.mark.parametrize('processor_class,extension', [(Unzip, '.zip'), (Untar, '.tar.gz')], ids=['Unzip', 'Untar'])
def test_unpacking(processor_class, extension, target_path, archive, members):
    """Tests the behaviour of processors for unpacking archives (Untar, Unzip)"""
    processor = processor_class(members=members, extract_dir=target_path)
    if target_path is None:
        target_path = archive + extension + processor.suffix
    with TemporaryDirectory() as path:
        path = Path(path)
        true_paths, expected_log = _unpacking_expected_paths_and_logs(archive, members, path / target_path, processor_class.__name__)
        pup = Pooch(path=path, base_url=BASEURL, registry=REGISTRY)
        with capture_log() as log_file:
            fnames = pup.fetch(archive + extension, processor=processor)
            assert set(fnames) == true_paths
            _check_logs(log_file, expected_log)
        for fname in fnames:
            check_tiny_data(fname)
        with capture_log() as log_file:
            fnames = pup.fetch(archive + extension, processor=processor)
            assert set(fnames) == true_paths
            _check_logs(log_file, [])
        for fname in fnames:
            check_tiny_data(fname)