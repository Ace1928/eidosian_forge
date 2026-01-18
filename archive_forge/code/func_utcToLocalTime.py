from pyparsing import *
import datetime,time
def utcToLocalTime(tokens):
    utctime = datetime.datetime.strptime('%(date)s %(time)s' % tokens, '%Y/%m/%d %H:%M:%S')
    localtime = utctime - datetime.timedelta(0, time.timezone, 0)
    tokens['utcdate'], tokens['utctime'] = (tokens['date'], tokens['time'])
    tokens['localdate'], tokens['localtime'] = str(localtime).split()
    del tokens['date']
    del tokens['time']