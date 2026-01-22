import os
import subprocess
import pyudev
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
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


def get_device_info(device_id):
    try:
        device_info = {}
        result = subprocess.run(
            ["adb", "-s", device_id, "shell", "getprop"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.stderr:
            raise Exception(result.stderr.decode())
        props = result.stdout.decode().strip().split("\n")
        for prop in props:
            key, value = prop.split(": ")
            device_info[key.strip()] = value.strip()
        return device_info
    except Exception as e:
        print(f"Error getting device info: {e}")
        return {}


def install_tools(device_info):
    try:
        if (
            "ro.product.manufacturer" in device_info
            and device_info["ro.product.manufacturer"] == "Samsung"
        ):
            os.system("sudo apt install -y samsung-tools")
        os.system("sudo apt install -y adb fastboot")
    except Exception as e:
        print(f"Error installing tools: {e}")


def user_prompt(message):
    return input(f"{message} (y/n): ").lower() == "y"


def setup_device(device_id, custom_rom_path=None):
    try:
        info = get_device_info(device_id)
        install_tools(info)
        print(f"Preparing device {device_id}...")
        if user_prompt("Would you like to unlock the bootloader?"):
            os.system(f"adb -s {device_id} reboot bootloader")
            os.system(f"fastboot -s {device_id} oem unlock")
        if user_prompt("Would you like to install Magisk?"):
            os.system(f"adb -s {device_id} install magisk.apk")
        if custom_rom_path:
            print(f"Installing custom ROM from {custom_rom_path}...")
            os.system(f"adb -s {device_id} sideload {custom_rom_path}")
    except Exception as e:
        print(f"Error setting up device: {e}")


class RootingTool(QtWidgets.QMainWindow):
    def __init__(self):
        super(RootingTool, self).__init__()
        uic.loadUi(
            "/home/lloyd/Downloads/PythonScripts/rootingtool/rooting_tool.ui", self
        )

        # Connect buttons to functions
        self.detectDevicesButton.clicked.connect(self.detect_devices)
        self.setupDeviceButton.clicked.connect(self.setup_device)
        self.selectRomButton.clicked.connect(self.select_rom)

        # Connect help menu action
        self.actionDocumentation.triggered.connect(self.show_documentation)

        # Initialize custom ROM path
        self.custom_rom_path = None

    def detect_devices(self):
        self.usbDevicesList.clear()
        usb_devices = detect_usb_devices()
        for device in usb_devices:
            self.usbDevicesList.addItem(
                f"Model: {device['model']}, Manufacturer: {device['manufacturer']}, Serial: {device['serial']}"
            )

        self.adbDevicesList.clear()
        adb_devices = detect_adb_devices()
        for device in adb_devices:
            self.adbDevicesList.addItem(device)

    def select_rom(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom ROM",
            "",
            "ZIP Files (*.zip);;All Files (*)",
            options=options,
        )
        if file_name:
            self.custom_rom_path = file_name
            self.romFileLabel.setText(f"Selected ROM: {os.path.basename(file_name)}")

    def setup_device(self):
        selected_device = self.adbDevicesList.currentItem().text()
        setup_device(selected_device, self.custom_rom_path)

    def show_documentation(self):
        doc_message = (
            "Rooting Tool Documentation\n\n"
            "Detect Devices: Detects and lists all connected USB and ADB devices.\n"
            "Select Custom ROM: Opens a file dialog to select a custom ROM file for installation.\n"
            "Setup Device: Sets up the selected ADB device, optionally installing the selected custom ROM.\n"
            "\nSteps to use the tool:\n"
            "1. Click 'Detect Devices' to list connected devices.\n"
            "2. (Optional) Click 'Select Custom ROM' to choose a ROM file.\n"
            "3. Select an ADB device from the list and click 'Setup Device' to begin the setup process.\n"
            "\nFor further assistance, refer to the official documentation or contact support."
        )
        QMessageBox.information(self, "Documentation", doc_message)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = RootingTool()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
