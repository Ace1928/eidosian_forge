from decimal import Decimal
from boto.compat import filter, map
class GetReportListResult(ResponseElement):
    ReportInfo = ElementList()