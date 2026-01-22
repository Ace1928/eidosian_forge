import serial
from serial import (
from twisted.python.runtime import platform
class BaseSerialPort:
    """
    Base class for Windows and POSIX serial ports.

    @ivar _serialFactory: a pyserial C{serial.Serial} factory, used to create
        the instance stored in C{self._serial}. Overrideable to enable easier
        testing.

    @ivar _serial: a pyserial C{serial.Serial} instance used to manage the
        options on the serial port.
    """
    _serialFactory = serial.Serial

    def setBaudRate(self, baudrate):
        if hasattr(self._serial, 'setBaudrate'):
            self._serial.setBaudrate(baudrate)
        else:
            self._serial.setBaudRate(baudrate)

    def inWaiting(self):
        return self._serial.inWaiting()

    def flushInput(self):
        self._serial.flushInput()

    def flushOutput(self):
        self._serial.flushOutput()

    def sendBreak(self):
        self._serial.sendBreak()

    def getDSR(self):
        return self._serial.getDSR()

    def getCD(self):
        return self._serial.getCD()

    def getRI(self):
        return self._serial.getRI()

    def getCTS(self):
        return self._serial.getCTS()

    def setDTR(self, on=1):
        self._serial.setDTR(on)

    def setRTS(self, on=1):
        self._serial.setRTS(on)