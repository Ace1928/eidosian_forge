from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
class TestComplexFieldNames:
    """
    The tests in this class should verify that the datashape parser can handle
    field names that contain strange characters like spaces, quotes, and
    backslashes

    The idea is that any given input datashape should be recoverable once we
    have created the actual dshape object.

    This test suite is by no means complete, but it does handle some of the
    more common special cases (common special? oxymoron?)
    """

    def test_spaces_01(self):
        space_dshape = "{'Unique Key': ?int64}"
        assert space_dshape == str(dshape(space_dshape))

    def test_spaces_02(self):
        big_space_dshape = "{ 'Unique Key' : ?int64, 'Created Date' : string,\n'Closed Date' : string, Agency : string, 'Agency Name' : string,\n'Complaint Type' : string, Descriptor : string, 'Location Type' : string,\n'Incident Zip' : ?int64, 'Incident Address' : ?string, 'Street Name' : ?string,\n'Cross Street 1' : ?string, 'Cross Street 2' : ?string,\n'Intersection Street 1' : ?string, 'Intersection Street 2' : ?string,\n'Address Type' : string, City : string, Landmark : string,\n'Facility Type' : string, Status : string, 'Due Date' : string,\n'Resolution Action Updated Date' : string, 'Community Board' : string,\nBorough : string, 'X Coordinate (State Plane)' : ?int64,\n'Y Coordinate (State Plane)' : ?int64, 'Park Facility Name' : string,\n'Park Borough' : string, 'School Name' : string, 'School Number' : string,\n'School Region' : string, 'School Code' : string,\n'School Phone Number' : string, 'School Address' : string,\n'School City' : string, 'School State' : string, 'School Zip' : string,\n'School Not Found' : string, 'School or Citywide Complaint' : string,\n'Vehicle Type' : string, 'Taxi Company Borough' : string,\n'Taxi Pick Up Location' : string, 'Bridge Highway Name' : string,\n'Bridge Highway Direction' : string, 'Road Ramp' : string,\n'Bridge Highway Segment' : string, 'Garage Lot Name' : string,\n'Ferry Direction' : string, 'Ferry Terminal Name' : string,\nLatitude : ?float64, Longitude : ?float64, Location : string }"
        ds1 = dshape(big_space_dshape)
        ds2 = dshape(str(ds1))
        assert str(ds1) == str(ds2)

    def test_single_quotes_01(self):
        quotes_dshape = "{ 'field \\' with \\' quotes' : string }"
        ds1 = dshape(quotes_dshape)
        ds2 = dshape(str(ds1))
        assert str(ds1) == str(ds2)

    def test_double_quotes_01(self):
        quotes_dshape = '{ \'doublequote " field "\' : int64 }'
        ds1 = dshape(quotes_dshape)
        ds2 = dshape(str(ds1))
        assert str(ds1) == str(ds2)

    def test_multi_quotes_01(self):
        quotes_dshape = '{\n            \'field \\\' with \\\' quotes\' : string,\n            \'doublequote " field "\' : int64\n        }'
        ds1 = dshape(quotes_dshape)
        ds2 = dshape(str(ds1))
        assert str(ds1) == str(ds2)

    def test_mixed_quotes_01(self):
        quotes_dshape = '{\n            \'field " with \\\' quotes\' : string,\n            \'doublequote " field \\\'\' : int64\n        }'
        ds1 = dshape(quotes_dshape)
        ds2 = dshape(str(ds1))
        assert str(ds1) == str(ds2)

    def test_bad_02(self):
        bad_dshape = '{ Unique Key : int64}'
        with pytest.raises(error.DataShapeSyntaxError):
            dshape(bad_dshape)

    def test_bad_backslashes_01(self):
        """backslashes aren't allowed in datashapes according to the definitions
        in lexer.py as of 2014-10-02. This is probably an oversight that should
        be fixed.
        """
        backslash_dshape = "{ 'field with \\\\   backslashes' : int64 }"
        with pytest.raises(error.DataShapeSyntaxError):
            dshape(backslash_dshape)