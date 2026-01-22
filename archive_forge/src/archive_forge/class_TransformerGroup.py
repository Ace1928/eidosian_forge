import threading
import warnings
from abc import ABC, abstractmethod
from array import array
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Any, Optional, Union, overload
from pyproj import CRS
from pyproj._compat import cstrencode
from pyproj._crs import AreaOfUse, CoordinateOperation
from pyproj._datadir import _clear_proj_error
from pyproj._transformer import (  # noqa: F401 pylint: disable=unused-import
from pyproj.datadir import get_user_data_dir
from pyproj.enums import ProjVersion, TransformDirection, WktVersion
from pyproj.exceptions import ProjError
from pyproj.sync import _download_resource_file
from pyproj.utils import _convertback, _copytobuffer
class TransformerGroup(_TransformerGroup):
    """
    The TransformerGroup is a set of possible transformers from one CRS to another.

    .. versionadded:: 2.3.0

    .. warning:: CoordinateOperation and Transformer objects
                 returned are not thread-safe.

    From PROJ docs::

        The operations are sorted with the most relevant ones first: by
        descending area (intersection of the transformation area with the
        area of interest, or intersection of the transformation with the
        area of use of the CRS), and by increasing accuracy. Operations
        with unknown accuracy are sorted last, whatever their area.

    """

    def __init__(self, crs_from: Any, crs_to: Any, always_xy: bool=False, area_of_interest: Optional[AreaOfInterest]=None, authority: Optional[str]=None, accuracy: Optional[float]=None, allow_ballpark: bool=True, allow_superseded: bool=False) -> None:
        """Get all possible transformations from a :obj:`pyproj.crs.CRS`
        or input used to create one.

        .. versionadded:: 3.4.0 authority, accuracy, allow_ballpark
        .. versionadded:: 3.6.0 allow_superseded

        Parameters
        ----------
        crs_from: pyproj.crs.CRS or input used to create one
            Projection of input data.
        crs_to: pyproj.crs.CRS or input used to create one
            Projection of output data.
        always_xy: bool, default=False
            If true, the transform method will accept as input and return as output
            coordinates using the traditional GIS order, that is longitude, latitude
            for geographic CRS and easting, northing for most projected CRS.
        area_of_interest: :class:`.AreaOfInterest`, optional
            The area of interest to help order the transformations based on the
            best operation for the area.
        authority: str, optional
            When not specified, coordinate operations from any authority will be
            searched, with the restrictions set in the
            authority_to_authority_preference database table related to the
            authority of the source/target CRS themselves. If authority is set
            to “any”, then coordinate operations from any authority will be
            searched. If authority is a non-empty string different from "any",
            then coordinate operations will be searched only in that authority
            namespace (e.g. EPSG).
        accuracy: float, optional
            The minimum desired accuracy (in metres) of the candidate
            coordinate operations.
        allow_ballpark: bool, default=True
            Set to False to disallow the use of Ballpark transformation
            in the candidate coordinate operations. Default is to allow.
        allow_superseded: bool, default=False
            Set to True to allow the use of superseded (but not deprecated)
            transformations in the candidate coordinate operations. Default is
            to disallow.

        """
        super().__init__(CRS.from_user_input(crs_from)._crs, CRS.from_user_input(crs_to)._crs, always_xy=always_xy, area_of_interest=area_of_interest, authority=authority, accuracy=-1 if accuracy is None else accuracy, allow_ballpark=allow_ballpark, allow_superseded=allow_superseded)
        for iii, transformer in enumerate(self._transformers):
            self._transformers[iii] = Transformer(TransformerUnsafe(transformer))

    @property
    def transformers(self) -> list['Transformer']:
        """
        list[:obj:`Transformer`]:
            List of available :obj:`Transformer`
            associated with the transformation.
        """
        return self._transformers

    @property
    def unavailable_operations(self) -> list[CoordinateOperation]:
        """
        list[:obj:`pyproj.crs.CoordinateOperation`]:
            List of :obj:`pyproj.crs.CoordinateOperation` that are not
            available due to missing grids.
        """
        return self._unavailable_operations

    @property
    def best_available(self) -> bool:
        """
        bool: If True, the best possible transformer is available.
        """
        return self._best_available

    def download_grids(self, directory: Optional[Union[str, Path]]=None, open_license: bool=True, verbose: bool=False) -> None:
        """
        .. versionadded:: 3.0.0

        Download missing grids that can be downloaded automatically.

        .. warning:: There are cases where the URL to download the grid is missing.
                     In those cases, you can enable enable
                     :ref:`debugging-internal-proj` and perform a
                     transformation. The logs will show the grids PROJ searches for.

        Parameters
        ----------
        directory: str or Path, optional
            The directory to download the grids to.
            Defaults to :func:`pyproj.datadir.get_user_data_dir`
        open_license: bool, default=True
            If True, will only download grids with an open license.
        verbose: bool, default=False
            If True, will print information about grids downloaded.
        """
        if directory is None:
            directory = get_user_data_dir(True)
        for unavailable_operation in self.unavailable_operations:
            for grid in unavailable_operation.grids:
                if not grid.available and grid.url.endswith(grid.short_name) and grid.direct_download and (grid.open_license or not open_license):
                    _download_resource_file(file_url=grid.url, short_name=grid.short_name, directory=directory, verbose=verbose)
                elif not grid.available and verbose:
                    warnings.warn(f'Skipped: {grid}')

    def __repr__(self) -> str:
        return f'<TransformerGroup: best_available={self.best_available}>\n- transformers: {len(self.transformers)}\n- unavailable_operations: {len(self.unavailable_operations)}'