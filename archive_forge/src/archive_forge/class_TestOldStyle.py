from __future__ import print_function
import sys
import pytest
class TestOldStyle:

    def test_single_load(self):
        d = get_yaml().load(single_doc)
        print(d)
        print(type(d[0]))
        assert d == single_data

    def test_single_load_no_arg(self):
        with pytest.raises(TypeError):
            assert get_yaml().load() == single_data

    def test_multi_load(self):
        data = list(get_yaml().load_all(multi_doc))
        assert data == multi_doc_data

    def test_single_dump(self, capsys):
        get_yaml().dump(single_data, sys.stdout)
        out, err = capsys.readouterr()
        assert out == single_doc

    def test_multi_dump(self, capsys):
        yaml = get_yaml()
        yaml.explicit_start = True
        yaml.dump_all(multi_doc_data, sys.stdout)
        out, err = capsys.readouterr()
        assert out == multi_doc