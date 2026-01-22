import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
class LongitudeFormatter(_PlateCarreeFormatter):
    """Tick formatter for a longitude axis."""

    def __init__(self, direction_label=True, zero_direction_label=False, dateline_direction_label=False, degree_symbol='°', number_format='g', transform_precision=1e-08, dms=False, minute_symbol='′', second_symbol='″', seconds_number_format='g', auto_hide=True, decimal_point=None, cardinal_labels=None):
        """
        Create a formatter for longitudes.

        When bound to an axis, the axis must be part of an axes defined
        on a rectangular projection (e.g. Plate Carree, Mercator).

        Parameters
        ----------
        direction_label: optional
            If *True* a direction label (E or W) will be drawn next to
            longitude labels. If *False* then these
            labels will not be drawn. Defaults to *True* (draw direction
            labels).
        zero_direction_label: optional
            If *True* a direction label (E or W) will be drawn next to
            longitude labels with the value 0. If *False* then these
            labels will not be drawn. Defaults to *False* (no direction
            labels).
        dateline_direction_label: optional
            If *True* a direction label (E or W) will be drawn next to
            longitude labels with the value 180. If *False* then these
            labels will not be drawn. Defaults to *False* (no direction
            labels).
        degree_symbol: optional
            The symbol used to represent degrees. Defaults to '°'.
        number_format: optional
            Format string to represent the latitude values when `dms`
            is set to False. Defaults to 'g'.
        transform_precision: optional
            Sets the precision (in degrees) to which transformed tick
            values are rounded. The default is 1e-7, and should be
            suitable for most use cases. To control the appearance of
            tick labels use the *number_format* keyword.
        dms: bool, optional
            Whether or not formatting as degrees-minutes-seconds and not
            as decimal degrees.
        minute_symbol: str, optional
            The character(s) used to represent the minute symbol.
        second_symbol: str, optional
            The character(s) used to represent the second symbol.
        seconds_number_format: optional
            Format string to represent the "seconds" component of the latitude
            values. Defaults to 'g'.
        auto_hide: bool, optional
            Auto-hide degrees or minutes when redundant.
        decimal_point: bool, optional
            Decimal point character. If not provided and
            ``mpl.rcParams['axes.formatter.use_locale'] == True``,
            the locale decimal point is used.
        cardinal_labels: dict, optional
            A dictionary with "west" and/or "east" keys to replace west and
            east cardinal labels, which defaults to "W" and "E".

        Note
        ----
            A formatter can only be used for one axis. A new formatter
            must be created for every axis that needs formatted labels.

        Examples
        --------
        Label longitudes from -180 to 180 on a Plate Carree projection
        with a central longitude of 0::

            ax = plt.axes(projection=PlateCarree())
            ax.set_global()
            ax.set_xticks([-180, -120, -60, 0, 60, 120, 180],
                          crs=ccrs.PlateCarree())
            lon_formatter = LongitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)

        Label longitudes from 0 to 360 on a Plate Carree projection
        with a central longitude of 180::

            ax = plt.axes(projection=PlateCarree(central_longitude=180))
            ax.set_global()
            ax.set_xticks([0, 60, 120, 180, 240, 300, 360],
                          crs=ccrs.PlateCarree())
            lon_formatter = LongitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)


        When not bound to an axis::

            lon_formatter = LongitudeFormatter()
            ticks = [0, 60, 120, 180, 240, 300, 360]
            lon_formatter.set_locs(ticks)
            labels = [lon_formatter(value) for value in ticks]
        """
        super().__init__(direction_label=direction_label, degree_symbol=degree_symbol, number_format=number_format, transform_precision=transform_precision, dms=dms, minute_symbol=minute_symbol, second_symbol=second_symbol, seconds_number_format=seconds_number_format, auto_hide=auto_hide, decimal_point=decimal_point, cardinal_labels=cardinal_labels)
        self._zero_direction_labels = zero_direction_label
        self._dateline_direction_labels = dateline_direction_label

    def _apply_transform(self, value, target_proj, source_crs):
        return target_proj.transform_point(value, 0, source_crs)[0]

    @classmethod
    def _fix_lons(cls, lons):
        if isinstance(lons, list):
            return [cls._fix_lons(lon) for lon in lons]
        p180 = lons == 180
        m180 = lons == -180
        lons = (lons + 180) % 360 - 180
        for mp180, value in [(m180, -180), (p180, 180)]:
            if np.any(mp180):
                if isinstance(lons, np.ndarray):
                    lons = np.where(mp180, value, lons)
                else:
                    lons = value
        return lons

    def set_locs(self, locs):
        _PlateCarreeFormatter.set_locs(self, self._fix_lons(locs))

    def _format_degrees(self, deg):
        return _PlateCarreeFormatter._format_degrees(self, self._fix_lons(deg))

    def _hemisphere(self, value, value_source_crs):
        value = self._fix_lons(value)
        if value < 0:
            hemisphere = self._cardinal_labels.get('west', 'W')
        elif value > 0:
            hemisphere = self._cardinal_labels.get('east', 'E')
        else:
            hemisphere = ''
        if value == 0 and self._zero_direction_labels:
            if value_source_crs < 0:
                hemisphere = self._cardinal_labels.get('east', 'E')
            else:
                hemisphere = self._cardinal_labels.get('west', 'W')
        if value in (-180, 180) and (not self._dateline_direction_labels):
            hemisphere = ''
        return hemisphere