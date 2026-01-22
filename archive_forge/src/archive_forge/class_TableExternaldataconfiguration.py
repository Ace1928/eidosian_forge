from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TableExternaldataconfiguration(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'autodetect': self.request.get('autodetect'), u'compression': self.request.get('compression'), u'ignoreUnknownValues': self.request.get('ignore_unknown_values'), u'maxBadRecords': self.request.get('max_bad_records'), u'sourceFormat': self.request.get('source_format'), u'sourceUris': self.request.get('source_uris'), u'schema': TableSchema(self.request.get('schema', {}), self.module).to_request(), u'googleSheetsOptions': TableGooglesheetsoptions(self.request.get('google_sheets_options', {}), self.module).to_request(), u'csvOptions': TableCsvoptions(self.request.get('csv_options', {}), self.module).to_request(), u'bigtableOptions': TableBigtableoptions(self.request.get('bigtable_options', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'autodetect': self.request.get(u'autodetect'), u'compression': self.request.get(u'compression'), u'ignoreUnknownValues': self.request.get(u'ignoreUnknownValues'), u'maxBadRecords': self.request.get(u'maxBadRecords'), u'sourceFormat': self.request.get(u'sourceFormat'), u'sourceUris': self.request.get(u'sourceUris'), u'schema': TableSchema(self.request.get(u'schema', {}), self.module).from_response(), u'googleSheetsOptions': TableGooglesheetsoptions(self.request.get(u'googleSheetsOptions', {}), self.module).from_response(), u'csvOptions': TableCsvoptions(self.request.get(u'csvOptions', {}), self.module).from_response(), u'bigtableOptions': TableBigtableoptions(self.request.get(u'bigtableOptions', {}), self.module).from_response()})