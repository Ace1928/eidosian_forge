from typing import Any, Dict, Optional
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
def _format_weather_info(self, location: str, w: Any) -> str:
    detailed_status = w.detailed_status
    wind = w.wind()
    humidity = w.humidity
    temperature = w.temperature('celsius')
    rain = w.rain
    heat_index = w.heat_index
    clouds = w.clouds
    return f'In {location}, the current weather is as follows:\nDetailed status: {detailed_status}\nWind speed: {wind['speed']} m/s, direction: {wind['deg']}°\nHumidity: {humidity}%\nTemperature: \n  - Current: {temperature['temp']}°C\n  - High: {temperature['temp_max']}°C\n  - Low: {temperature['temp_min']}°C\n  - Feels like: {temperature['feels_like']}°C\nRain: {rain}\nHeat index: {heat_index}\nCloud cover: {clouds}%'