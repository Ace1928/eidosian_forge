from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
class PermissionType(enum.Enum):
    ACCESSIBILITY_EVENTS = 'accessibilityEvents'
    AUDIO_CAPTURE = 'audioCapture'
    BACKGROUND_SYNC = 'backgroundSync'
    BACKGROUND_FETCH = 'backgroundFetch'
    CAPTURED_SURFACE_CONTROL = 'capturedSurfaceControl'
    CLIPBOARD_READ_WRITE = 'clipboardReadWrite'
    CLIPBOARD_SANITIZED_WRITE = 'clipboardSanitizedWrite'
    DISPLAY_CAPTURE = 'displayCapture'
    DURABLE_STORAGE = 'durableStorage'
    FLASH = 'flash'
    GEOLOCATION = 'geolocation'
    IDLE_DETECTION = 'idleDetection'
    LOCAL_FONTS = 'localFonts'
    MIDI = 'midi'
    MIDI_SYSEX = 'midiSysex'
    NFC = 'nfc'
    NOTIFICATIONS = 'notifications'
    PAYMENT_HANDLER = 'paymentHandler'
    PERIODIC_BACKGROUND_SYNC = 'periodicBackgroundSync'
    PROTECTED_MEDIA_IDENTIFIER = 'protectedMediaIdentifier'
    SENSORS = 'sensors'
    STORAGE_ACCESS = 'storageAccess'
    TOP_LEVEL_STORAGE_ACCESS = 'topLevelStorageAccess'
    VIDEO_CAPTURE = 'videoCapture'
    VIDEO_CAPTURE_PAN_TILT_ZOOM = 'videoCapturePanTiltZoom'
    WAKE_LOCK_SCREEN = 'wakeLockScreen'
    WAKE_LOCK_SYSTEM = 'wakeLockSystem'
    WINDOW_MANAGEMENT = 'windowManagement'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)