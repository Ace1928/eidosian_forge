def test_parse_filename_no_ext():
    from ase.io.formats import parse_filename
    filename, index = parse_filename('path.to/filename')
    assert filename == 'path.to/filename'
    assert index is None