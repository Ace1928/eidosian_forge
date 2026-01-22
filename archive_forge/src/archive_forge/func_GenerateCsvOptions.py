from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.command_lib.iam import iam_util
def GenerateCsvOptions(args):
    return dataplex_api.GetMessageModule().GoogleCloudDataplexV1ZoneDiscoverySpecCsvOptions(delimiter=args.csv_delimiter, disableTypeInference=args.csv_disable_type_inference, encoding=args.csv_encoding, headerRows=args.csv_header_rows)