from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_spaces_02(self):
    big_space_dshape = "{ 'Unique Key' : ?int64, 'Created Date' : string,\n'Closed Date' : string, Agency : string, 'Agency Name' : string,\n'Complaint Type' : string, Descriptor : string, 'Location Type' : string,\n'Incident Zip' : ?int64, 'Incident Address' : ?string, 'Street Name' : ?string,\n'Cross Street 1' : ?string, 'Cross Street 2' : ?string,\n'Intersection Street 1' : ?string, 'Intersection Street 2' : ?string,\n'Address Type' : string, City : string, Landmark : string,\n'Facility Type' : string, Status : string, 'Due Date' : string,\n'Resolution Action Updated Date' : string, 'Community Board' : string,\nBorough : string, 'X Coordinate (State Plane)' : ?int64,\n'Y Coordinate (State Plane)' : ?int64, 'Park Facility Name' : string,\n'Park Borough' : string, 'School Name' : string, 'School Number' : string,\n'School Region' : string, 'School Code' : string,\n'School Phone Number' : string, 'School Address' : string,\n'School City' : string, 'School State' : string, 'School Zip' : string,\n'School Not Found' : string, 'School or Citywide Complaint' : string,\n'Vehicle Type' : string, 'Taxi Company Borough' : string,\n'Taxi Pick Up Location' : string, 'Bridge Highway Name' : string,\n'Bridge Highway Direction' : string, 'Road Ramp' : string,\n'Bridge Highway Segment' : string, 'Garage Lot Name' : string,\n'Ferry Direction' : string, 'Ferry Terminal Name' : string,\nLatitude : ?float64, Longitude : ?float64, Location : string }"
    ds1 = dshape(big_space_dshape)
    ds2 = dshape(str(ds1))
    assert str(ds1) == str(ds2)