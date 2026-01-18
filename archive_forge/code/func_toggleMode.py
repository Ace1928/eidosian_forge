from math import pi, sin
from struct import pack
from PySide2.QtCore import QByteArray, QIODevice, Qt, QTimer, qWarning
from PySide2.QtMultimedia import (QAudio, QAudioDeviceInfo, QAudioFormat,
from PySide2.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
def toggleMode(self):
    self.m_pullTimer.stop()
    self.m_audioOutput.stop()
    if self.m_pullMode:
        self.m_modeButton.setText(self.PULL_MODE_LABEL)
        self.m_output = self.m_audioOutput.start()
        self.m_pullMode = False
        self.m_pullTimer.start(20)
    else:
        self.m_modeButton.setText(self.PUSH_MODE_LABEL)
        self.m_pullMode = True
        self.m_audioOutput.start(self.m_generator)
    self.m_suspendResumeButton.setText(self.SUSPEND_LABEL)