from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def _telnet_process_subnegotiation(self, suboption):
    """Process subnegotiation, the data between IAC SB and IAC SE."""
    if suboption[0:1] == COM_PORT_OPTION:
        if self.logger:
            self.logger.debug('received COM_PORT_OPTION: {!r}'.format(suboption))
        if suboption[1:2] == SET_BAUDRATE:
            backup = self.serial.baudrate
            try:
                baudrate, = struct.unpack(b'!I', suboption[2:6])
                if baudrate != 0:
                    self.serial.baudrate = baudrate
            except ValueError as e:
                if self.logger:
                    self.logger.error('failed to set baud rate: {}'.format(e))
                self.serial.baudrate = backup
            else:
                if self.logger:
                    self.logger.info('{} baud rate: {}'.format('set' if baudrate else 'get', self.serial.baudrate))
            self.rfc2217_send_subnegotiation(SERVER_SET_BAUDRATE, struct.pack(b'!I', self.serial.baudrate))
        elif suboption[1:2] == SET_DATASIZE:
            backup = self.serial.bytesize
            try:
                datasize, = struct.unpack(b'!B', suboption[2:3])
                if datasize != 0:
                    self.serial.bytesize = datasize
            except ValueError as e:
                if self.logger:
                    self.logger.error('failed to set data size: {}'.format(e))
                self.serial.bytesize = backup
            else:
                if self.logger:
                    self.logger.info('{} data size: {}'.format('set' if datasize else 'get', self.serial.bytesize))
            self.rfc2217_send_subnegotiation(SERVER_SET_DATASIZE, struct.pack(b'!B', self.serial.bytesize))
        elif suboption[1:2] == SET_PARITY:
            backup = self.serial.parity
            try:
                parity = struct.unpack(b'!B', suboption[2:3])[0]
                if parity != 0:
                    self.serial.parity = RFC2217_REVERSE_PARITY_MAP[parity]
            except ValueError as e:
                if self.logger:
                    self.logger.error('failed to set parity: {}'.format(e))
                self.serial.parity = backup
            else:
                if self.logger:
                    self.logger.info('{} parity: {}'.format('set' if parity else 'get', self.serial.parity))
            self.rfc2217_send_subnegotiation(SERVER_SET_PARITY, struct.pack(b'!B', RFC2217_PARITY_MAP[self.serial.parity]))
        elif suboption[1:2] == SET_STOPSIZE:
            backup = self.serial.stopbits
            try:
                stopbits = struct.unpack(b'!B', suboption[2:3])[0]
                if stopbits != 0:
                    self.serial.stopbits = RFC2217_REVERSE_STOPBIT_MAP[stopbits]
            except ValueError as e:
                if self.logger:
                    self.logger.error('failed to set stop bits: {}'.format(e))
                self.serial.stopbits = backup
            else:
                if self.logger:
                    self.logger.info('{} stop bits: {}'.format('set' if stopbits else 'get', self.serial.stopbits))
            self.rfc2217_send_subnegotiation(SERVER_SET_STOPSIZE, struct.pack(b'!B', RFC2217_STOPBIT_MAP[self.serial.stopbits]))
        elif suboption[1:2] == SET_CONTROL:
            if suboption[2:3] == SET_CONTROL_REQ_FLOW_SETTING:
                if self.serial.xonxoff:
                    self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_USE_SW_FLOW_CONTROL)
                elif self.serial.rtscts:
                    self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_USE_HW_FLOW_CONTROL)
                else:
                    self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_USE_NO_FLOW_CONTROL)
            elif suboption[2:3] == SET_CONTROL_USE_NO_FLOW_CONTROL:
                self.serial.xonxoff = False
                self.serial.rtscts = False
                if self.logger:
                    self.logger.info('changed flow control to None')
                self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_USE_NO_FLOW_CONTROL)
            elif suboption[2:3] == SET_CONTROL_USE_SW_FLOW_CONTROL:
                self.serial.xonxoff = True
                if self.logger:
                    self.logger.info('changed flow control to XON/XOFF')
                self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_USE_SW_FLOW_CONTROL)
            elif suboption[2:3] == SET_CONTROL_USE_HW_FLOW_CONTROL:
                self.serial.rtscts = True
                if self.logger:
                    self.logger.info('changed flow control to RTS/CTS')
                self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_USE_HW_FLOW_CONTROL)
            elif suboption[2:3] == SET_CONTROL_REQ_BREAK_STATE:
                if self.logger:
                    self.logger.warning('requested break state - not implemented')
                pass
            elif suboption[2:3] == SET_CONTROL_BREAK_ON:
                self.serial.break_condition = True
                if self.logger:
                    self.logger.info('changed BREAK to active')
                self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_BREAK_ON)
            elif suboption[2:3] == SET_CONTROL_BREAK_OFF:
                self.serial.break_condition = False
                if self.logger:
                    self.logger.info('changed BREAK to inactive')
                self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_BREAK_OFF)
            elif suboption[2:3] == SET_CONTROL_REQ_DTR:
                if self.logger:
                    self.logger.warning('requested DTR state - not implemented')
                pass
            elif suboption[2:3] == SET_CONTROL_DTR_ON:
                self.serial.dtr = True
                if self.logger:
                    self.logger.info('changed DTR to active')
                self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_DTR_ON)
            elif suboption[2:3] == SET_CONTROL_DTR_OFF:
                self.serial.dtr = False
                if self.logger:
                    self.logger.info('changed DTR to inactive')
                self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_DTR_OFF)
            elif suboption[2:3] == SET_CONTROL_REQ_RTS:
                if self.logger:
                    self.logger.warning('requested RTS state - not implemented')
                pass
            elif suboption[2:3] == SET_CONTROL_RTS_ON:
                self.serial.rts = True
                if self.logger:
                    self.logger.info('changed RTS to active')
                self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_RTS_ON)
            elif suboption[2:3] == SET_CONTROL_RTS_OFF:
                self.serial.rts = False
                if self.logger:
                    self.logger.info('changed RTS to inactive')
                self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_RTS_OFF)
        elif suboption[1:2] == NOTIFY_LINESTATE:
            self.rfc2217_send_subnegotiation(SERVER_NOTIFY_LINESTATE, to_bytes([0]))
        elif suboption[1:2] == NOTIFY_MODEMSTATE:
            if self.logger:
                self.logger.info('request for modem state')
            self.check_modem_lines(force_notification=True)
        elif suboption[1:2] == FLOWCONTROL_SUSPEND:
            if self.logger:
                self.logger.info('suspend')
            self._remote_suspend_flow = True
        elif suboption[1:2] == FLOWCONTROL_RESUME:
            if self.logger:
                self.logger.info('resume')
            self._remote_suspend_flow = False
        elif suboption[1:2] == SET_LINESTATE_MASK:
            self.linstate_mask = ord(suboption[2:3])
            if self.logger:
                self.logger.info('line state mask: 0x{:02x}'.format(self.linstate_mask))
        elif suboption[1:2] == SET_MODEMSTATE_MASK:
            self.modemstate_mask = ord(suboption[2:3])
            if self.logger:
                self.logger.info('modem state mask: 0x{:02x}'.format(self.modemstate_mask))
        elif suboption[1:2] == PURGE_DATA:
            if suboption[2:3] == PURGE_RECEIVE_BUFFER:
                self.serial.reset_input_buffer()
                if self.logger:
                    self.logger.info('purge in')
                self.rfc2217_send_subnegotiation(SERVER_PURGE_DATA, PURGE_RECEIVE_BUFFER)
            elif suboption[2:3] == PURGE_TRANSMIT_BUFFER:
                self.serial.reset_output_buffer()
                if self.logger:
                    self.logger.info('purge out')
                self.rfc2217_send_subnegotiation(SERVER_PURGE_DATA, PURGE_TRANSMIT_BUFFER)
            elif suboption[2:3] == PURGE_BOTH_BUFFERS:
                self.serial.reset_input_buffer()
                self.serial.reset_output_buffer()
                if self.logger:
                    self.logger.info('purge both')
                self.rfc2217_send_subnegotiation(SERVER_PURGE_DATA, PURGE_BOTH_BUFFERS)
            elif self.logger:
                self.logger.error('undefined PURGE_DATA: {!r}'.format(list(suboption[2:])))
        elif self.logger:
            self.logger.error('undefined COM_PORT_OPTION: {!r}'.format(list(suboption[1:])))
    elif self.logger:
        self.logger.warning('unknown subnegotiation: {!r}'.format(suboption))