import unittest
from prov.model import ProvDocument
from prov.tests.utility import RoundTripTestCase
from prov.tests.test_model import (
import os
from glob import glob
import logging
from prov.tests import examples
import prov.model as pm
import rdflib as rl
from rdflib.compare import graph_diff
from io import BytesIO, StringIO
class TestStatementsBase2(TestStatementsBase):

    @unittest.expectedFailure
    def test_scruffy_end_1(self):
        TestStatementsBase.test_scruffy_end_1(self)

    @unittest.expectedFailure
    def test_scruffy_end_2(self):
        TestStatementsBase.test_scruffy_end_2(self)

    @unittest.expectedFailure
    def test_scruffy_end_3(self):
        TestStatementsBase.test_scruffy_end_3(self)

    @unittest.expectedFailure
    def test_scruffy_end_4(self):
        TestStatementsBase.test_scruffy_end_4(self)

    @unittest.expectedFailure
    def test_scruffy_generation_1(self):
        TestStatementsBase.test_scruffy_generation_1(self)

    @unittest.expectedFailure
    def test_scruffy_generation_2(self):
        TestStatementsBase.test_scruffy_generation_2(self)

    @unittest.expectedFailure
    def test_scruffy_invalidation_1(self):
        TestStatementsBase.test_scruffy_invalidation_1(self)

    @unittest.expectedFailure
    def test_scruffy_invalidation_2(self):
        TestStatementsBase.test_scruffy_invalidation_2(self)

    @unittest.expectedFailure
    def test_scruffy_start_1(self):
        TestStatementsBase.test_scruffy_start_1(self)

    @unittest.expectedFailure
    def test_scruffy_start_2(self):
        TestStatementsBase.test_scruffy_start_2(self)

    @unittest.expectedFailure
    def test_scruffy_start_3(self):
        TestStatementsBase.test_scruffy_start_3(self)

    @unittest.expectedFailure
    def test_scruffy_start_4(self):
        TestStatementsBase.test_scruffy_start_4(self)

    @unittest.expectedFailure
    def test_scruffy_usage_1(self):
        TestStatementsBase.test_scruffy_usage_1(self)

    @unittest.expectedFailure
    def test_scruffy_usage_2(self):
        TestStatementsBase.test_scruffy_usage_2(self)