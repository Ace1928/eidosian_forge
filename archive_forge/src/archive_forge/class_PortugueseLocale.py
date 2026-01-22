import sys
from math import trunc
from typing import (
class PortugueseLocale(Locale):
    names = ['pt', 'pt-pt']
    past = 'há {0}'
    future = 'em {0}'
    and_word = 'e'
    timeframes = {'now': 'agora', 'second': 'um segundo', 'seconds': '{0} segundos', 'minute': 'um minuto', 'minutes': '{0} minutos', 'hour': 'uma hora', 'hours': '{0} horas', 'day': 'um dia', 'days': '{0} dias', 'week': 'uma semana', 'weeks': '{0} semanas', 'month': 'um mês', 'months': '{0} meses', 'year': 'um ano', 'years': '{0} anos'}
    month_names = ['', 'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
    month_abbreviations = ['', 'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    day_names = ['', 'Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
    day_abbreviations = ['', 'Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']