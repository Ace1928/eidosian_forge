import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def travel_plug_to_eu_standard(adaptee):
    if adaptee.mode == 'Europe':
        return ex.TravelPlugToEUStandard(adaptee=adaptee)
    else:
        return None