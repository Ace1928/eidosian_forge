def test_parse_filename_bad_slice():
    from ase.io.formats import parse_filename
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        filename, index = parse_filename('path.to/filename@s:4')
        assert filename == 'path.to/filename'
        assert len(w) == 1
        assert 'Can not parse index' in str(w[-1].message)