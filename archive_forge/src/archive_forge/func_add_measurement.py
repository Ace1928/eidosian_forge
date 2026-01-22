import six
from pycadf import attachment
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import identifier
from pycadf import measurement
from pycadf import reason
from pycadf import reporterstep
from pycadf import resource
from pycadf import tag
from pycadf import timestamp
def add_measurement(self, measure_val):
    """Add a measurement value

        :param measure_val: Measurement data type to be added to Event
        """
    if measure_val is not None and isinstance(measure_val, measurement.Measurement):
        if measure_val.is_valid():
            if not hasattr(self, EVENT_KEYNAME_MEASUREMENTS):
                setattr(self, EVENT_KEYNAME_MEASUREMENTS, list())
            measurements = getattr(self, EVENT_KEYNAME_MEASUREMENTS)
            measurements.append(measure_val)
        else:
            raise ValueError('Invalid measurement')
    else:
        raise ValueError('Invalid measurement. Value must be a Measurement')