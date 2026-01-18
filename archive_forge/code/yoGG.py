import os
import subprocess
import pyudev
from PyQt5 import QtWidgets, uic, QFileDialog, QMessageBox
import sys


def detect_usb_devices():
    try:
        context = pyudev.Context()
        devices = []
        for device in context.list_devices(subsystem="usb", DEVTYPE="usb_device"):
            if "ID_MODEL" in device:
                devices.append(
                    {
                        "model": device.get("ID_MODEL"),
                        "manufacturer": device.get("ID_VENDOR"),
                        "serial": device.get("ID_SERIAL_SHORT"),
                    }
                )
        return devices
    except Exception as e:
        print(f"Error detecting USB devices: {e}")
        return []


def detect_adb_devices():
    try:
        result = subprocess.run(
            ["adb", "devices"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.stderr:
            raise Exception(result.stderr.decode())
        devices = result.stdout.decode().strip().split("\n")[1:]
        connected_devices = [
            line.split("\t")[0] for line in devices if "device" in line
        ]
        return connected_devices
    except Exception as e:
        print(f"Error detecting ADB devices: {e}")
        return []
