import csv
import datetime
import os
def _get_date(row, column):
    return convert_date(row[column]) if column in row else None