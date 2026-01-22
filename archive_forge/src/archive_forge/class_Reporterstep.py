from pycadf import cadftype
from pycadf import identifier
from pycadf import resource
from pycadf import timestamp
class Reporterstep(cadftype.CADFAbstractType):
    role = cadftype.ValidatorDescriptor(REPORTERSTEP_KEYNAME_ROLE, lambda x: cadftype.is_valid_reporter_role(x))
    reporter = cadftype.ValidatorDescriptor(REPORTERSTEP_KEYNAME_REPORTER, lambda x: isinstance(x, resource.Resource) and x.is_valid())
    reporterId = cadftype.ValidatorDescriptor(REPORTERSTEP_KEYNAME_REPORTERID, lambda x: identifier.is_valid(x))
    reporterTime = cadftype.ValidatorDescriptor(REPORTERSTEP_KEYNAME_REPORTERTIME, lambda x: timestamp.is_valid(x))

    def __init__(self, role=cadftype.REPORTER_ROLE_MODIFIER, reporterTime=None, reporter=None, reporterId=None):
        """Create ReporterStep data type

        :param role: optional role of Reporterstep. Defaults to 'modifier'
        :param reporterTime: utc time of Reporterstep.
        :param reporter: CADF Resource of reporter
        :param reporterId: id of CADF resource for reporter
        """
        setattr(self, REPORTERSTEP_KEYNAME_ROLE, role)
        if reporterTime is not None:
            setattr(self, REPORTERSTEP_KEYNAME_REPORTERTIME, reporterTime)
        if reporter is not None:
            setattr(self, REPORTERSTEP_KEYNAME_REPORTER, reporter)
        if reporterId is not None:
            setattr(self, REPORTERSTEP_KEYNAME_REPORTERID, reporterId)

    def is_valid(self):
        """Validation to ensure Reporterstep required attributes are set.
        """
        return self._isset(REPORTERSTEP_KEYNAME_ROLE) and self._isset(REPORTERSTEP_KEYNAME_REPORTER) ^ self._isset(REPORTERSTEP_KEYNAME_REPORTERID)