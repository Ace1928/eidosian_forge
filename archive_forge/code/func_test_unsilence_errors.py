import threading
import h5py
def test_unsilence_errors(tmp_path, capfd):
    """Check that HDF5 errors can be muted/unmuted from h5py"""
    filename = tmp_path / 'test.h5'
    try:
        h5py._errors.unsilence_errors()
        _access_not_existing_object(filename)
        captured = capfd.readouterr()
        assert captured.err != ''
        assert captured.out == ''
    finally:
        h5py._errors.silence_errors()
    _access_not_existing_object(filename)
    captured = capfd.readouterr()
    assert captured.err == ''
    assert captured.out == ''