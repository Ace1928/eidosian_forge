import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
class TestNestedToRecord:

    def test_flat_stays_flat(self):
        recs = [{'flat1': 1, 'flat2': 2}, {'flat3': 3, 'flat2': 4}]
        result = nested_to_record(recs)
        expected = recs
        assert result == expected

    def test_one_level_deep_flattens(self):
        data = {'flat1': 1, 'dict1': {'c': 1, 'd': 2}}
        result = nested_to_record(data)
        expected = {'dict1.c': 1, 'dict1.d': 2, 'flat1': 1}
        assert result == expected

    def test_nested_flattens(self):
        data = {'flat1': 1, 'dict1': {'c': 1, 'd': 2}, 'nested': {'e': {'c': 1, 'd': 2}, 'd': 2}}
        result = nested_to_record(data)
        expected = {'dict1.c': 1, 'dict1.d': 2, 'flat1': 1, 'nested.d': 2, 'nested.e.c': 1, 'nested.e.d': 2}
        assert result == expected

    def test_json_normalize_errors(self, missing_metadata):
        msg = "Key 'name' not found. To replace missing values of 'name' with np.nan, pass in errors='ignore'"
        with pytest.raises(KeyError, match=msg):
            json_normalize(data=missing_metadata, record_path='addresses', meta='name', errors='raise')

    def test_missing_meta(self, missing_metadata):
        result = json_normalize(data=missing_metadata, record_path='addresses', meta='name', errors='ignore')
        ex_data = [[9562, 'Morris St.', 'Massillon', 'OH', 44646, 'Alice'], [8449, 'Spring St.', 'Elizabethton', 'TN', 37643, np.nan]]
        columns = ['number', 'street', 'city', 'state', 'zip', 'name']
        expected = DataFrame(ex_data, columns=columns)
        tm.assert_frame_equal(result, expected)

    def test_missing_nested_meta(self):
        data = {'meta': 'foo', 'nested_meta': None, 'value': [{'rec': 1}, {'rec': 2}]}
        result = json_normalize(data, record_path='value', meta=['meta', ['nested_meta', 'leaf']], errors='ignore')
        ex_data = [[1, 'foo', np.nan], [2, 'foo', np.nan]]
        columns = ['rec', 'meta', 'nested_meta.leaf']
        expected = DataFrame(ex_data, columns=columns).astype({'nested_meta.leaf': object})
        tm.assert_frame_equal(result, expected)
        with pytest.raises(KeyError, match="'leaf' not found"):
            json_normalize(data, record_path='value', meta=['meta', ['nested_meta', 'leaf']], errors='raise')

    def test_missing_meta_multilevel_record_path_errors_raise(self, missing_metadata):
        msg = "Key 'name' not found. To replace missing values of 'name' with np.nan, pass in errors='ignore'"
        with pytest.raises(KeyError, match=msg):
            json_normalize(data=missing_metadata, record_path=['previous_residences', 'cities'], meta='name', errors='raise')

    def test_missing_meta_multilevel_record_path_errors_ignore(self, missing_metadata):
        result = json_normalize(data=missing_metadata, record_path=['previous_residences', 'cities'], meta='name', errors='ignore')
        ex_data = [['Foo York City', 'Alice'], ['Barmingham', np.nan]]
        columns = ['city_name', 'name']
        expected = DataFrame(ex_data, columns=columns)
        tm.assert_frame_equal(result, expected)

    def test_donot_drop_nonevalues(self):
        data = [{'info': None, 'author_name': {'first': 'Smith', 'last_name': 'Appleseed'}}, {'info': {'created_at': '11/08/1993', 'last_updated': '26/05/2012'}, 'author_name': {'first': 'Jane', 'last_name': 'Doe'}}]
        result = nested_to_record(data)
        expected = [{'info': None, 'author_name.first': 'Smith', 'author_name.last_name': 'Appleseed'}, {'author_name.first': 'Jane', 'author_name.last_name': 'Doe', 'info.created_at': '11/08/1993', 'info.last_updated': '26/05/2012'}]
        assert result == expected

    def test_nonetype_top_level_bottom_level(self):
        data = {'id': None, 'location': {'country': {'state': {'id': None, 'town.info': {'id': None, 'region': None, 'x': 49.151580810546875, 'y': -33.148521423339844, 'z': 27.572303771972656}}}}}
        result = nested_to_record(data)
        expected = {'id': None, 'location.country.state.id': None, 'location.country.state.town.info.id': None, 'location.country.state.town.info.region': None, 'location.country.state.town.info.x': 49.151580810546875, 'location.country.state.town.info.y': -33.148521423339844, 'location.country.state.town.info.z': 27.572303771972656}
        assert result == expected

    def test_nonetype_multiple_levels(self):
        data = {'id': None, 'location': {'id': None, 'country': {'id': None, 'state': {'id': None, 'town.info': {'region': None, 'x': 49.151580810546875, 'y': -33.148521423339844, 'z': 27.572303771972656}}}}}
        result = nested_to_record(data)
        expected = {'id': None, 'location.id': None, 'location.country.id': None, 'location.country.state.id': None, 'location.country.state.town.info.region': None, 'location.country.state.town.info.x': 49.151580810546875, 'location.country.state.town.info.y': -33.148521423339844, 'location.country.state.town.info.z': 27.572303771972656}
        assert result == expected

    @pytest.mark.parametrize('max_level, expected', [(None, [{'CreatedBy.Name': 'User001', 'Lookup.TextField': 'Some text', 'Lookup.UserField.Id': 'ID001', 'Lookup.UserField.Name': 'Name001', 'Image.a': 'b'}]), (0, [{'CreatedBy': {'Name': 'User001'}, 'Lookup': {'TextField': 'Some text', 'UserField': {'Id': 'ID001', 'Name': 'Name001'}}, 'Image': {'a': 'b'}}]), (1, [{'CreatedBy.Name': 'User001', 'Lookup.TextField': 'Some text', 'Lookup.UserField': {'Id': 'ID001', 'Name': 'Name001'}, 'Image.a': 'b'}])])
    def test_with_max_level(self, max_level, expected, max_level_test_input_data):
        output = nested_to_record(max_level_test_input_data, max_level=max_level)
        assert output == expected

    def test_with_large_max_level(self):
        max_level = 100
        input_data = [{'CreatedBy': {'user': {'name': {'firstname': 'Leo', 'LastName': 'Thomson'}, 'family_tree': {'father': {'name': 'Father001', 'father': {'Name': 'Father002', 'father': {'name': 'Father003', 'father': {'Name': 'Father004'}}}}}}}}]
        expected = [{'CreatedBy.user.name.firstname': 'Leo', 'CreatedBy.user.name.LastName': 'Thomson', 'CreatedBy.user.family_tree.father.name': 'Father001', 'CreatedBy.user.family_tree.father.father.Name': 'Father002', 'CreatedBy.user.family_tree.father.father.father.name': 'Father003', 'CreatedBy.user.family_tree.father.father.father.father.Name': 'Father004'}]
        output = nested_to_record(input_data, max_level=max_level)
        assert output == expected

    def test_series_non_zero_index(self):
        data = {0: {'id': 1, 'name': 'Foo', 'elements': {'a': 1}}, 1: {'id': 2, 'name': 'Bar', 'elements': {'b': 2}}, 2: {'id': 3, 'name': 'Baz', 'elements': {'c': 3}}}
        s = Series(data)
        s.index = [1, 2, 3]
        result = json_normalize(s)
        expected = DataFrame({'id': [1, 2, 3], 'name': ['Foo', 'Bar', 'Baz'], 'elements.a': [1.0, np.nan, np.nan], 'elements.b': [np.nan, 2.0, np.nan], 'elements.c': [np.nan, np.nan, 3.0]})
        tm.assert_frame_equal(result, expected)