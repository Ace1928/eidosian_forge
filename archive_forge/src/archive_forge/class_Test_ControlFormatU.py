import unittest
import logging
from os_ken.lib.packet import llc
class Test_ControlFormatU(Test_ControlFormatI):
    msg = llc.llc(llc.SAP_BPDU, llc.SAP_BPDU, llc.ControlFormatU())