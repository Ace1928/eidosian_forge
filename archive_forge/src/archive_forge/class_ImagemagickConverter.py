import subprocess
import sys
from subprocess import PIPE, CalledProcessError
from typing import Any, Dict
import sphinx
from sphinx.application import Sphinx
from sphinx.errors import ExtensionError
from sphinx.locale import __
from sphinx.transforms.post_transforms.images import ImageConverter
from sphinx.util import logging
class ImagemagickConverter(ImageConverter):
    conversion_rules = [('image/svg+xml', 'image/png'), ('image/gif', 'image/png'), ('application/pdf', 'image/png'), ('application/illustrator', 'image/png')]

    def is_available(self) -> bool:
        """Confirms the converter is available or not."""
        try:
            args = [self.config.image_converter, '-version']
            logger.debug('Invoking %r ...', args)
            subprocess.run(args, stdout=PIPE, stderr=PIPE, check=True)
            return True
        except OSError as exc:
            logger.warning(__("Unable to run the image conversion command %r. 'sphinx.ext.imgconverter' requires ImageMagick by default. Ensure it is installed, or set the 'image_converter' option to a custom conversion command.\n\nTraceback: %s"), self.config.image_converter, exc)
            return False
        except CalledProcessError as exc:
            logger.warning(__('convert exited with error:\n[stderr]\n%r\n[stdout]\n%r'), exc.stderr, exc.stdout)
            return False

    def convert(self, _from: str, _to: str) -> bool:
        """Converts the image to expected one."""
        try:
            _from += '[0]'
            args = [self.config.image_converter] + self.config.image_converter_args + [_from, _to]
            logger.debug('Invoking %r ...', args)
            subprocess.run(args, stdout=PIPE, stderr=PIPE, check=True)
            return True
        except OSError:
            logger.warning(__('convert command %r cannot be run, check the image_converter setting'), self.config.image_converter)
            return False
        except CalledProcessError as exc:
            raise ExtensionError(__('convert exited with error:\n[stderr]\n%r\n[stdout]\n%r') % (exc.stderr, exc.stdout)) from exc