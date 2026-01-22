import sys
from PySide2.QtCore import qApp, QPointF, Qt
from PySide2.QtGui import QColor, QPainter, QPalette
from PySide2.QtWidgets import (QApplication, QMainWindow, QSizePolicy,
from PySide2.QtCharts import QtCharts
from ui_themewidget import Ui_ThemeWidgetForm as ui
from random import random, uniform
class ThemeWidget(QWidget):

    def __init__(self, parent):
        QWidget.__init__(self, parent)
        self.charts = []
        self.ui = ui()
        self.list_count = 3
        self.value_max = 10
        self.value_count = 7
        self.data_table = self.generate_random_data(self.list_count, self.value_max, self.value_count)
        self.ui.setupUi(self)
        self.populate_themebox()
        self.populate_animationbox()
        self.populate_legendbox()
        chart_view = QtCharts.QChartView(self.create_areachart())
        self.ui.gridLayout.addWidget(chart_view, 1, 0)
        self.charts.append(chart_view)
        chart_view = QtCharts.QChartView(self.createPieChart())
        chart_view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.ui.gridLayout.addWidget(chart_view, 1, 1)
        self.charts.append(chart_view)
        chart_view = QtCharts.QChartView(self.createLineChart())
        self.ui.gridLayout.addWidget(chart_view, 1, 2)
        self.charts.append(chart_view)
        chart_view = QtCharts.QChartView(self.createBarChart())
        self.ui.gridLayout.addWidget(chart_view, 2, 0)
        self.charts.append(chart_view)
        chart_view = QtCharts.QChartView(self.createSplineChart())
        self.ui.gridLayout.addWidget(chart_view, 2, 1)
        self.charts.append(chart_view)
        chart_view = QtCharts.QChartView(self.create_scatterchart())
        self.ui.gridLayout.addWidget(chart_view, 2, 2)
        self.charts.append(chart_view)
        self.ui.antialiasCheckBox.setChecked(True)
        pal = qApp.palette()
        pal.setColor(QPalette.Window, QColor(15790320))
        pal.setColor(QPalette.WindowText, QColor(4210756))
        qApp.setPalette(pal)
        self.updateUI()

    def generate_random_data(self, list_count, value_max, value_count):
        data_table = []
        for i in range(list_count):
            data_list = []
            y_value = 0
            for j in range(value_count):
                constant = value_max / float(value_count)
                y_value += uniform(0, constant)
                x_value = (j + random()) * constant
                value = QPointF(x_value, y_value)
                label = 'Slice {}: {}'.format(i, j)
                data_list.append((value, label))
            data_table.append(data_list)
        return data_table

    def populate_themebox(self):
        theme = self.ui.themeComboBox
        theme.addItem('Light', QtCharts.QChart.ChartThemeLight)
        theme.addItem('Blue Cerulean', QtCharts.QChart.ChartThemeBlueCerulean)
        theme.addItem('Dark', QtCharts.QChart.ChartThemeDark)
        theme.addItem('Brown Sand', QtCharts.QChart.ChartThemeBrownSand)
        theme.addItem('Blue NCS', QtCharts.QChart.ChartThemeBlueNcs)
        theme.addItem('High Contrast', QtCharts.QChart.ChartThemeHighContrast)
        theme.addItem('Blue Icy', QtCharts.QChart.ChartThemeBlueIcy)
        theme.addItem('Qt', QtCharts.QChart.ChartThemeQt)

    def populate_animationbox(self):
        animated = self.ui.animatedComboBox
        animated.addItem('No Animations', QtCharts.QChart.NoAnimation)
        animated.addItem('GridAxis Animations', QtCharts.QChart.GridAxisAnimations)
        animated.addItem('Series Animations', QtCharts.QChart.SeriesAnimations)
        animated.addItem('All Animations', QtCharts.QChart.AllAnimations)

    def populate_legendbox(self):
        legend = self.ui.legendComboBox
        legend.addItem('No Legend ', 0)
        legend.addItem('Legend Top', Qt.AlignTop)
        legend.addItem('Legend Bottom', Qt.AlignBottom)
        legend.addItem('Legend Left', Qt.AlignLeft)
        legend.addItem('Legend Right', Qt.AlignRight)

    def create_areachart(self):
        chart = QtCharts.QChart()
        chart.setTitle('Area Chart')
        lower_series = None
        name = 'Series '
        for i in range(len(self.data_table)):
            upper_series = QtCharts.QLineSeries(chart)
            for j in range(len(self.data_table[i])):
                data = self.data_table[i][j]
                if lower_series:
                    points = lower_series.pointsVector()
                    y_value = points[i].y() + data[0].y()
                    upper_series.append(QPointF(j, y_value))
                else:
                    upper_series.append(QPointF(j, data[0].y()))
            area = QtCharts.QAreaSeries(upper_series, lower_series)
            area.setName('{}{}'.format(name, i))
            chart.addSeries(area)
            lower_series = upper_series
        chart.createDefaultAxes()
        chart.axisX().setRange(0, self.value_count - 1)
        chart.axisY().setRange(0, self.value_max)
        chart.axisY().setLabelFormat('%.1f  ')
        return chart

    def createBarChart(self):
        chart = QtCharts.QChart()
        chart.setTitle('Bar chart')
        series = QtCharts.QStackedBarSeries(chart)
        for i in range(len(self.data_table)):
            barset = QtCharts.QBarSet('Bar set {}'.format(i))
            for data in self.data_table[i]:
                barset.append(data[0].y())
            series.append(barset)
        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.axisY().setRange(0, self.value_max * 2)
        chart.axisY().setLabelFormat('%.1f  ')
        return chart

    def createLineChart(self):
        chart = QtCharts.QChart()
        chart.setTitle('Line chart')
        name = 'Series '
        for i, lst in enumerate(self.data_table):
            series = QtCharts.QLineSeries(chart)
            for data in lst:
                series.append(data[0])
            series.setName('{}{}'.format(name, i))
            chart.addSeries(series)
        chart.createDefaultAxes()
        chart.axisX().setRange(0, self.value_max)
        chart.axisY().setRange(0, self.value_count)
        chart.axisY().setLabelFormat('%.1f  ')
        return chart

    def createPieChart(self):
        chart = QtCharts.QChart()
        chart.setTitle('Pie chart')
        series = QtCharts.QPieSeries(chart)
        for data in self.data_table[0]:
            slc = series.append(data[1], data[0].y())
            if data == self.data_table[0][0]:
                slc.setLabelVisible()
                slc.setExploded()
                slc.setExplodeDistanceFactor(0.5)
        series.setPieSize(0.4)
        chart.addSeries(series)
        return chart

    def createSplineChart(self):
        chart = QtCharts.QChart()
        chart.setTitle('Spline chart')
        name = 'Series '
        for i, lst in enumerate(self.data_table):
            series = QtCharts.QSplineSeries(chart)
            for data in lst:
                series.append(data[0])
            series.setName('{}{}'.format(name, i))
            chart.addSeries(series)
        chart.createDefaultAxes()
        chart.axisX().setRange(0, self.value_max)
        chart.axisY().setRange(0, self.value_count)
        chart.axisY().setLabelFormat('%.1f  ')
        return chart

    def create_scatterchart(self):
        chart = QtCharts.QChart()
        chart.setTitle('Scatter chart')
        name = 'Series '
        for i, lst in enumerate(self.data_table):
            series = QtCharts.QScatterSeries(chart)
            for data in lst:
                series.append(data[0])
            series.setName('{}{}'.format(name, i))
            chart.addSeries(series)
        chart.createDefaultAxes()
        chart.axisX().setRange(0, self.value_max)
        chart.axisY().setRange(0, self.value_count)
        chart.axisY().setLabelFormat('%.1f  ')
        return chart

    def updateUI(self):

        def set_colors(window_color, text_color):
            pal = self.window().palette()
            pal.setColor(QPalette.Window, window_color)
            pal.setColor(QPalette.WindowText, text_color)
            self.window().setPalette(pal)
        idx = self.ui.themeComboBox.currentIndex()
        theme = self.ui.themeComboBox.itemData(idx)
        if len(self.charts):
            chart_theme = self.charts[0].chart().theme()
            if chart_theme != theme:
                for chart_view in self.charts:
                    if theme == 0:
                        theme_name = QtCharts.QChart.ChartThemeLight
                    elif theme == 1:
                        theme_name = QtCharts.QChart.ChartThemeBlueCerulean
                    elif theme == 2:
                        theme_name = QtCharts.QChart.ChartThemeDark
                    elif theme == 3:
                        theme_name = QtCharts.QChart.ChartThemeBrownSand
                    elif theme == 4:
                        theme_name = QtCharts.QChart.ChartThemeBlueNcs
                    elif theme == 5:
                        theme_name = QtCharts.QChart.ChartThemeHighContrast
                    elif theme == 6:
                        theme_name = QtCharts.QChart.ChartThemeBlueIcy
                    elif theme == 7:
                        theme_name = QtCharts.QChart.ChartThemeQt
                    else:
                        theme_name = QtCharts.QChart.ChartThemeLight
                    chart_view.chart().setTheme(theme_name)
                if theme == QtCharts.QChart.ChartThemeLight:
                    set_colors(QColor(15790320), QColor(4210756))
                elif theme == QtCharts.QChart.ChartThemeDark:
                    set_colors(QColor(1184280), QColor(14079702))
                elif theme == QtCharts.QChart.ChartThemeBlueCerulean:
                    set_colors(QColor(4211530), QColor(14079702))
                elif theme == QtCharts.QChart.ChartThemeBrownSand:
                    set_colors(QColor(10389861), QColor(4210756))
                elif theme == QtCharts.QChart.ChartThemeBlueNcs:
                    set_colors(QColor(101306), QColor(4210756))
                elif theme == QtCharts.QChart.ChartThemeHighContrast:
                    set_colors(QColor(16755459), QColor(1579032))
                elif theme == QtCharts.QChart.ChartThemeBlueIcy:
                    set_colors(QColor(13559792), QColor(4210756))
                else:
                    set_colors(QColor(15790320), QColor(4210756))
        checked = self.ui.antialiasCheckBox.isChecked()
        for chart in self.charts:
            chart.setRenderHint(QPainter.Antialiasing, checked)
        idx = self.ui.animatedComboBox.currentIndex()
        options = self.ui.animatedComboBox.itemData(idx)
        if len(self.charts):
            chart = self.charts[0].chart()
            animation_options = chart.animationOptions()
            if animation_options != options:
                for chart_view in self.charts:
                    options_name = QtCharts.QChart.NoAnimation
                    if options == 0:
                        options_name = QtCharts.QChart.NoAnimation
                    elif options == 1:
                        options_name = QtCharts.QChart.GridAxisAnimations
                    elif options == 2:
                        options_name = QtCharts.QChart.SeriesAnimations
                    elif options == 3:
                        options_name = QtCharts.QChart.AllAnimations
                    chart_view.chart().setAnimationOptions(options_name)
        idx = self.ui.legendComboBox.currentIndex()
        alignment = self.ui.legendComboBox.itemData(idx)
        if not alignment:
            for chart_view in self.charts:
                chart_view.chart().legend().hide()
        else:
            for chart_view in self.charts:
                alignment_name = Qt.AlignTop
                if alignment == 32:
                    alignment_name = Qt.AlignTop
                elif alignment == 64:
                    alignment_name = Qt.AlignBottom
                elif alignment == 1:
                    alignment_name = Qt.AlignLeft
                elif alignment == 2:
                    alignment_name = Qt.AlignRight
                chart_view.chart().legend().setAlignment(alignment_name)
                chart_view.chart().legend().show()