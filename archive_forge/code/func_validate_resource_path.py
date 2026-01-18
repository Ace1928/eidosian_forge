import os
from collections import abc
from functools import partial
from gi.repository import GLib, GObject, Gio
def validate_resource_path(path):
    """Raises GLib.Error in case the resource doesn't exist"""
    try:
        Gio.resources_get_info(path, Gio.ResourceLookupFlags.NONE)
    except GLib.Error:
        Gio.resources_lookup_data(path, Gio.ResourceLookupFlags.NONE)