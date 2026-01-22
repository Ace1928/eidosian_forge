from serial import EIGHTBITS, PARITY_NONE, STOPBITS_ONE
from twisted.internet import abstract, fdesc
from twisted.internet.serialport import BaseSerialPort

        Called when the serial port disconnects.

        Will call C{connectionLost} on the protocol that is handling the
        serial data.
        