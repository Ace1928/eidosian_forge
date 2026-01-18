import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
def test_send_scsi_inquiry_v2(self):
    self._ctypes_mocker.stop()
    fake_port_wwn = fc_struct.HBA_WWN()
    fake_remote_port_wwn = fc_struct.HBA_WWN()
    fake_fcp_lun = 11
    fake_cdb_byte_1 = 1
    fake_cdb_byte_2 = 128
    fake_resp = bytearray(range(200))
    fake_sense_data = bytearray(range(200)[::-1])
    fake_scsi_status = 5

    def mock_run(func, hba_handle, port_wwn_struct, remote_port_wwn_struct, fcp_lun, cdb_byte1, cdb_byte2, p_resp_buff, p_resp_buff_sz, p_scsi_status, p_sense_buff, p_sense_buff_sz):
        self.assertEqual(fc_utils.hbaapi.HBA_ScsiInquiryV2, func)
        self.assertEqual(mock.sentinel.hba_handle, hba_handle)
        self.assertEqual(fake_port_wwn, port_wwn_struct)
        self.assertEqual(fake_remote_port_wwn, remote_port_wwn_struct)
        self.assertEqual(fake_fcp_lun, fcp_lun.value)
        self.assertEqual(fake_cdb_byte_1, cdb_byte1.value)
        self.assertEqual(fake_cdb_byte_2, cdb_byte2.value)
        resp_buff_sz = ctypes.cast(p_resp_buff_sz, ctypes.POINTER(ctypes.c_uint32)).contents
        sense_buff_sz = ctypes.cast(p_sense_buff_sz, ctypes.POINTER(ctypes.c_uint32)).contents
        scsi_status = ctypes.cast(p_scsi_status, ctypes.POINTER(ctypes.c_ubyte)).contents
        self.assertEqual(fc_utils.SCSI_INQ_BUFF_SZ, resp_buff_sz.value)
        self.assertEqual(fc_utils.SENSE_BUFF_SZ, sense_buff_sz.value)
        resp_buff_type = ctypes.c_ubyte * resp_buff_sz.value
        sense_buff_type = ctypes.c_ubyte * sense_buff_sz.value
        resp_buff = ctypes.cast(p_resp_buff, ctypes.POINTER(resp_buff_type)).contents
        sense_buff = ctypes.cast(p_sense_buff, ctypes.POINTER(sense_buff_type)).contents
        resp_buff[:len(fake_resp)] = fake_resp
        sense_buff[:len(fake_sense_data)] = fake_sense_data
        resp_buff_sz.value = len(fake_resp)
        sense_buff_sz.value = len(fake_sense_data)
        scsi_status.value = fake_scsi_status
    self._mock_run.side_effect = mock_run
    resp_buff = self._fc_utils._send_scsi_inquiry_v2(mock.sentinel.hba_handle, fake_port_wwn, fake_remote_port_wwn, fake_fcp_lun, fake_cdb_byte_1, fake_cdb_byte_2)
    self.assertEqual(fake_resp, bytearray(resp_buff[:len(fake_resp)]))