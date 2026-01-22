from decimal import Decimal
from boto.compat import filter, map
class GetReportRequestListResult(RequestReportResult):
    ReportRequestInfo = ElementList()