import os, sys, time
import sqlite3
import binascii
import re
import csv
from the Manifold censuses in csv format to build sqlite

    Given a csv of manifolds data and a connection to a sqlite database,
    insert the data into a new table. If the csv file is in a subdirectory
    of the csv directory csv_dir, it is given by sub_dir.
    