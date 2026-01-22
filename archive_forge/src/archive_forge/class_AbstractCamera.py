import os
import platform
import sys
import warnings
from abc import ABC, abstractmethod
from pygame import error
class AbstractCamera(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """ """

    @abstractmethod
    def start(self):
        """ """

    @abstractmethod
    def stop(self):
        """ """

    @abstractmethod
    def get_size(self):
        """ """

    @abstractmethod
    def query_image(self):
        """ """

    @abstractmethod
    def get_image(self, dest_surf=None):
        """ """

    @abstractmethod
    def get_raw(self):
        """ """