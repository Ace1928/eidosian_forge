import random
from PySide2.QtCore import (Signal, QByteArray, QDataStream, QIODevice,
from PySide2.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel,
from PySide2.QtNetwork import (QHostAddress, QNetworkInterface, QTcpServer,
PySide2 port of the network/threadedfortuneserver example from Qt v5.x, originating from PyQt