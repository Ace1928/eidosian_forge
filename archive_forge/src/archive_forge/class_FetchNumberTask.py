import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow import task
class FetchNumberTask(task.Task):
    """Task that fetches number from phone book."""
    default_provides = 'number'

    def execute(self, person):
        print('Fetching number for %s.' % person)
        return PHONE_BOOK[person.lower()]