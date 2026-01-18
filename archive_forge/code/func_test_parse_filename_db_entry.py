def test_parse_filename_db_entry():
    from ase.io.formats import parse_filename
    filename, index = parse_filename('path.to/filename.db@anything')
    assert filename == 'path.to/filename.db'
    assert index == 'anything'