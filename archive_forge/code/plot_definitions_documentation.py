
heatmap/v1
{
    "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
    "padding": 5,
    "width": 500,
    "height": 500,
    "data":
      {
        "name": "${history-table:rows:x-axis,key}"
      },
    "title": {
      "text": {"value": ""}
    },
    "encoding": {
      "x": {
        "field": "x_axis",
        "type": "nominal",
        "axis": { "title": "" }
      },
      "y": {
        "field": "y_axis",
        "type": "nominal",
        "axis": { "title": "" }
      }
    },
    "layer": [
      {
        "mark": "rect",
        "encoding": {
          "color": {
            "field": "values",
            "type": "quantitative",
            "title": "Values",
            "scale": {
              "scheme": "tealblues"
            }
          }
        }
      },
      {
        "mark": "text",
        "encoding": {
          "text": {"field": "values", "type": "quantitative"}
        }
      }
    ]
}
