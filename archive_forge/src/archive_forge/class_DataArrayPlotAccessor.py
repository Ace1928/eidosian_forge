from __future__ import annotations
import functools
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, Literal, NoReturn, overload
import numpy as np
from xarray.plot import dataarray_plot, dataset_plot
class DataArrayPlotAccessor:
    """
    Enables use of xarray.plot functions as attributes on a DataArray.
    For example, DataArray.plot.imshow
    """
    _da: DataArray
    __slots__ = ('_da',)
    __doc__ = dataarray_plot.plot.__doc__

    def __init__(self, darray: DataArray) -> None:
        self._da = darray

    @functools.wraps(dataarray_plot.plot, assigned=('__doc__', '__annotations__'))
    def __call__(self, **kwargs) -> Any:
        return dataarray_plot.plot(self._da, **kwargs)

    @functools.wraps(dataarray_plot.hist)
    def hist(self, *args, **kwargs) -> tuple[np.ndarray, np.ndarray, BarContainer | Polygon]:
        return dataarray_plot.hist(self._da, *args, **kwargs)

    @overload
    def line(self, *args: Any, row: None=None, col: None=None, figsize: Iterable[float] | None=None, aspect: AspectOptions=None, size: float | None=None, ax: Axes | None=None, hue: Hashable | None=None, x: Hashable | None=None, y: Hashable | None=None, xincrease: bool | None=None, yincrease: bool | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, add_legend: bool=True, _labels: bool=True, **kwargs: Any) -> list[Line3D]:
        ...

    @overload
    def line(self, *args: Any, row: Hashable, col: Hashable | None=None, figsize: Iterable[float] | None=None, aspect: AspectOptions=None, size: float | None=None, ax: Axes | None=None, hue: Hashable | None=None, x: Hashable | None=None, y: Hashable | None=None, xincrease: bool | None=None, yincrease: bool | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, add_legend: bool=True, _labels: bool=True, **kwargs: Any) -> FacetGrid[DataArray]:
        ...

    @overload
    def line(self, *args: Any, row: Hashable | None=None, col: Hashable, figsize: Iterable[float] | None=None, aspect: AspectOptions=None, size: float | None=None, ax: Axes | None=None, hue: Hashable | None=None, x: Hashable | None=None, y: Hashable | None=None, xincrease: bool | None=None, yincrease: bool | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, add_legend: bool=True, _labels: bool=True, **kwargs: Any) -> FacetGrid[DataArray]:
        ...

    @functools.wraps(dataarray_plot.line, assigned=('__doc__',))
    def line(self, *args, **kwargs) -> list[Line3D] | FacetGrid[DataArray]:
        return dataarray_plot.line(self._da, *args, **kwargs)

    @overload
    def step(self, *args: Any, where: Literal['pre', 'post', 'mid']='pre', drawstyle: str | None=None, ds: str | None=None, row: None=None, col: None=None, **kwargs: Any) -> list[Line3D]:
        ...

    @overload
    def step(self, *args: Any, where: Literal['pre', 'post', 'mid']='pre', drawstyle: str | None=None, ds: str | None=None, row: Hashable, col: Hashable | None=None, **kwargs: Any) -> FacetGrid[DataArray]:
        ...

    @overload
    def step(self, *args: Any, where: Literal['pre', 'post', 'mid']='pre', drawstyle: str | None=None, ds: str | None=None, row: Hashable | None=None, col: Hashable, **kwargs: Any) -> FacetGrid[DataArray]:
        ...

    @functools.wraps(dataarray_plot.step, assigned=('__doc__',))
    def step(self, *args, **kwargs) -> list[Line3D] | FacetGrid[DataArray]:
        return dataarray_plot.step(self._da, *args, **kwargs)

    @overload
    def scatter(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, z: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, markersize: Hashable | None=None, linewidth: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: float | None=None, ax: Axes | None=None, row: None=None, col: None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_legend: bool | None=None, add_colorbar: bool | None=None, add_labels: bool | Iterable[bool]=True, add_title: bool=True, subplot_kws: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, cmap=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, extend=None, levels=None, **kwargs) -> PathCollection:
        ...

    @overload
    def scatter(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, z: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, markersize: Hashable | None=None, linewidth: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: float | None=None, ax: Axes | None=None, row: Hashable | None=None, col: Hashable, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_legend: bool | None=None, add_colorbar: bool | None=None, add_labels: bool | Iterable[bool]=True, add_title: bool=True, subplot_kws: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, cmap=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, extend=None, levels=None, **kwargs) -> FacetGrid[DataArray]:
        ...

    @overload
    def scatter(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, z: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, markersize: Hashable | None=None, linewidth: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: float | None=None, ax: Axes | None=None, row: Hashable, col: Hashable | None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_legend: bool | None=None, add_colorbar: bool | None=None, add_labels: bool | Iterable[bool]=True, add_title: bool=True, subplot_kws: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, cmap=None, vmin: float | None=None, vmax: float | None=None, norm: Normalize | None=None, extend=None, levels=None, **kwargs) -> FacetGrid[DataArray]:
        ...

    @functools.wraps(dataarray_plot.scatter, assigned=('__doc__',))
    def scatter(self, *args, **kwargs) -> PathCollection | FacetGrid[DataArray]:
        return dataarray_plot.scatter(self._da, *args, **kwargs)

    @overload
    def imshow(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: None=None, col: None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> AxesImage:
        ...

    @overload
    def imshow(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: Hashable | None=None, col: Hashable, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> FacetGrid[DataArray]:
        ...

    @overload
    def imshow(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: Hashable, col: Hashable | None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> FacetGrid[DataArray]:
        ...

    @functools.wraps(dataarray_plot.imshow, assigned=('__doc__',))
    def imshow(self, *args, **kwargs) -> AxesImage | FacetGrid[DataArray]:
        return dataarray_plot.imshow(self._da, *args, **kwargs)

    @overload
    def contour(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: None=None, col: None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> QuadContourSet:
        ...

    @overload
    def contour(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: Hashable | None=None, col: Hashable, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> FacetGrid[DataArray]:
        ...

    @overload
    def contour(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: Hashable, col: Hashable | None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> FacetGrid[DataArray]:
        ...

    @functools.wraps(dataarray_plot.contour, assigned=('__doc__',))
    def contour(self, *args, **kwargs) -> QuadContourSet | FacetGrid[DataArray]:
        return dataarray_plot.contour(self._da, *args, **kwargs)

    @overload
    def contourf(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: None=None, col: None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> QuadContourSet:
        ...

    @overload
    def contourf(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: Hashable | None=None, col: Hashable, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> FacetGrid[DataArray]:
        ...

    @overload
    def contourf(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: Hashable, col: Hashable | None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> FacetGrid:
        ...

    @functools.wraps(dataarray_plot.contourf, assigned=('__doc__',))
    def contourf(self, *args, **kwargs) -> QuadContourSet | FacetGrid[DataArray]:
        return dataarray_plot.contourf(self._da, *args, **kwargs)

    @overload
    def pcolormesh(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: None=None, col: None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> QuadMesh:
        ...

    @overload
    def pcolormesh(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: Hashable | None=None, col: Hashable, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> FacetGrid[DataArray]:
        ...

    @overload
    def pcolormesh(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: Hashable, col: Hashable | None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> FacetGrid[DataArray]:
        ...

    @functools.wraps(dataarray_plot.pcolormesh, assigned=('__doc__',))
    def pcolormesh(self, *args, **kwargs) -> QuadMesh | FacetGrid[DataArray]:
        return dataarray_plot.pcolormesh(self._da, *args, **kwargs)

    @overload
    def surface(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: None=None, col: None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> Poly3DCollection:
        ...

    @overload
    def surface(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: Hashable | None=None, col: Hashable, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> FacetGrid:
        ...

    @overload
    def surface(self, *args: Any, x: Hashable | None=None, y: Hashable | None=None, figsize: Iterable[float] | None=None, size: float | None=None, aspect: AspectOptions=None, ax: Axes | None=None, row: Hashable, col: Hashable | None=None, col_wrap: int | None=None, xincrease: bool | None=True, yincrease: bool | None=True, add_colorbar: bool | None=None, add_labels: bool=True, vmin: float | None=None, vmax: float | None=None, cmap=None, center=None, robust: bool=False, extend=None, levels=None, infer_intervals=None, colors=None, subplot_kws: dict[str, Any] | None=None, cbar_ax: Axes | None=None, cbar_kwargs: dict[str, Any] | None=None, xscale: ScaleOptions=None, yscale: ScaleOptions=None, xticks: ArrayLike | None=None, yticks: ArrayLike | None=None, xlim: tuple[float, float] | None=None, ylim: tuple[float, float] | None=None, norm: Normalize | None=None, **kwargs: Any) -> FacetGrid:
        ...

    @functools.wraps(dataarray_plot.surface, assigned=('__doc__',))
    def surface(self, *args, **kwargs) -> Poly3DCollection:
        return dataarray_plot.surface(self._da, *args, **kwargs)